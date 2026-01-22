from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from absl import app
from absl import flags
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
class Truncate(bigquery_command.BigqueryCmd):
    usage = 'bq truncate project_id:dataset[.table] [--timestamp] [--dry_run] [--overwrite] [--skip_fully_replicated_tables]\n'

    def __init__(self, name: str, fv: flags.FlagValues):
        super(Truncate, self).__init__(name, fv)
        flags.DEFINE_integer('timestamp', None, 'Optional timestamp to which table(s) will be truncated. Specified as milliseconds since epoch.', short_name='t', flag_values=fv)
        flags.DEFINE_boolean('dry_run', None, 'No-op that simply prints out information and the recommended timestamp without modifying tables or datasets.', flag_values=fv)
        flags.DEFINE_boolean('overwrite', False, 'Overwrite existing tables. Otherwise timestamp will be appended to all output table names.', flag_values=fv)
        flags.DEFINE_boolean('skip_fully_replicated_tables', True, 'Skip tables that are fully replicated (synced) and do not need to be truncated back to a point in time. This could result in datasets that have tables synchronized to different points in time, but will require less data to be re-loaded', short_name='s', flag_values=fv)
        self._ProcessCommandRc(fv)

    def RunWithArgs(self, identifier: str='') -> Optional[int]:
        """Truncates table/dataset/project to a particular timestamp.

    Examples:
      bq truncate project_id:dataset
      bq truncate --overwrite project_id:dataset --timestamp 123456789
      bq truncate --skip_fully_replicated_tables=false project_id:dataset
    """
        client = bq_cached_client.Client.Get()
        if identifier:
            reference = client.GetReference(identifier.strip())
        else:
            raise app.UsageError('Must specify one of project, dataset or table')
        self.truncated_table_count = 0
        self.skipped_table_count = 0
        self.failed_table_count = 0
        status = []
        if self.timestamp and (not self.dry_run):
            print('Truncating to user specified timestamp %s.(Not skipping fully replicated tables.)' % self.timestamp)
            if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
                all_tables = [reference]
            elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
                all_tables = list(map(lambda x: client.GetReference(x['id']), client.ListTables(reference, max_results=1000 * 1000)))
            for a_table in all_tables:
                try:
                    status.append(self._TruncateTable(a_table, str(self.timestamp), False))
                except bq_error.BigqueryError as e:
                    print(e)
                    status.append(self._formatOutputString(a_table, 'Failed'))
                    self.failed_table_count += 1
        else:
            if isinstance(reference, bq_id_utils.ApiClientHelper.TableReference):
                all_table_infos = self._GetTableInfo(reference)
            elif isinstance(reference, bq_id_utils.ApiClientHelper.DatasetReference):
                all_table_infos = self._GetTableInfosFromDataset(reference)
            try:
                recovery_timestamp = min(list(map(self._GetRecoveryTimestamp, all_table_infos)))
            except (ValueError, TypeError):
                recovery_timestamp = None
            if not recovery_timestamp:
                raise app.UsageError('Unable to figure out a recovery timestamp for %s. Exiting.' % reference)
            print('Recommended timestamp to truncate to is %s' % recovery_timestamp)
            for a_table in all_table_infos:
                try:
                    table_reference = bq_id_utils.ApiClientHelper.TableReference.Create(projectId=reference.projectId, datasetId=reference.datasetId, tableId=a_table['name'])
                    status.append(self._TruncateTable(table_reference, str(recovery_timestamp), a_table['fully_replicated']))
                except bq_error.BigqueryError as e:
                    print(e)
                    status.append(self._formatOutputString(table_reference, 'Failed'))
                    self.failed_table_count += 1
        print('%s tables truncated, %s tables failed to truncate, %s tables skipped' % (self.truncated_table_count, self.failed_table_count, self.skipped_table_count))
        print(*status, sep='\n')

    def _GetTableInfosFromDataset(self, dataset_reference: bq_id_utils.ApiClientHelper.DatasetReference):
        recovery_timestamp_for_dataset_query = 'SELECT\n  TABLE_NAME,\n  UNIX_MILLIS(replicated_time_at_remote_site),\n  CASE\n    WHEN last_update_time <= min_latest_replicated_time THEN TRUE\n  ELSE\n  FALSE\nEND\n  AS fully_replicated\nFROM (\n  SELECT\n    TABLE_NAME,\n    multi_site_info.last_update_time,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time DESC)[safe_OFFSET(1)] AS replicated_time_at_remote_site,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time ASC)[safe_OFFSET(0)] AS min_latest_replicated_time\n  FROM\n    %s.INFORMATION_SCHEMA.TABLES t,\n    t.multi_site_info.site_info\n  GROUP BY\n    1,\n    2)' % dataset_reference.datasetId
        return self._ReadTableInfo(recovery_timestamp_for_dataset_query, 1000 * 1000)

    def _GetTableInfo(self, table_reference: bq_id_utils.ApiClientHelper.TableReference):
        recovery_timestamp_for_table_query = "SELECT\n  TABLE_NAME,\n  UNIX_MILLIS(replicated_time_at_remote_site),\n  CASE\n    WHEN last_update_time <= min_latest_replicated_time THEN TRUE\n  ELSE\n  FALSE\nEND\n  AS fully_replicated\nFROM (\n  SELECT\n    TABLE_NAME,\n    multi_site_info.last_update_time,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time DESC)[safe_OFFSET(1)] AS replicated_time_at_remote_site,\n    ARRAY_AGG(site_info.latest_replicated_time\n    ORDER BY\n      latest_replicated_time ASC)[safe_OFFSET(0)] AS min_latest_replicated_time\n  FROM\n    %s.INFORMATION_SCHEMA.TABLES t,\n    t.multi_site_info.site_info\n  WHERE\n    TABLE_NAME = '%s'\n  GROUP BY\n    1,\n    2 )" % (table_reference.datasetId, table_reference.tableId)
        return self._ReadTableInfo(recovery_timestamp_for_table_query, row_count=1)

    def _GetRecoveryTimestamp(self, table_info) -> Optional[int]:
        return int(table_info['recovery_timestamp']) if table_info['recovery_timestamp'] else None

    def _ReadTableInfo(self, query: str, row_count: int):
        client = bq_cached_client.Client.Get()
        try:
            job = client.Query(query, use_legacy_sql=False)
        except bq_error.BigqueryError as e:
            if 'Name multi_site_info not found' in e.error['message']:
                raise app.UsageError('This functionality is not enabled for the current project.')
            else:
                raise e
        all_table_infos = []
        if not bq_client_utils.IsFailedJob(job):
            _, rows = client.ReadSchemaAndJobRows(job['jobReference'], start_row=0, max_rows=row_count)
            for i in range(len(rows)):
                table_info = {}
                table_info['name'] = rows[i][0]
                table_info['recovery_timestamp'] = rows[i][1]
                table_info['fully_replicated'] = rows[i][2] == 'true'
                all_table_infos.append(table_info)
            return all_table_infos

    def _formatOutputString(self, table_reference: bq_id_utils.ApiClientHelper.TableReference, status: str) -> str:
        return '%s %200s' % (table_reference, status)

    def _TruncateTable(self, table_reference: bq_id_utils.ApiClientHelper.TableReference, recovery_timestamp: str, is_fully_replicated: bool) -> str:
        client = bq_cached_client.Client.Get()
        kwds = {}
        if not self.overwrite:
            dest = bq_id_utils.ApiClientHelper.TableReference.Create(projectId=table_reference.projectId, datasetId=table_reference.datasetId, tableId='_'.join([table_reference.tableId, 'TRUNCATED_AT', recovery_timestamp]))
        else:
            dest = table_reference
        if self.skip_fully_replicated_tables and is_fully_replicated:
            self.skipped_table_count += 1
            return self._formatOutputString(table_reference, 'Fully replicated...Skipped')
        if self.dry_run:
            return self._formatOutputString(dest, 'will be Truncated@%s' % recovery_timestamp)
        kwds = {'write_disposition': 'WRITE_TRUNCATE', 'ignore_already_exists': 'False', 'operation_type': 'COPY'}
        if FLAGS.location:
            kwds['location'] = FLAGS.location
        source_table = client.GetTableReference('%s@%s' % (table_reference, recovery_timestamp))
        job_ref = ' '
        try:
            job = client.CopyTable([source_table], dest, **kwds)
            if job is None:
                self.failed_table_count += 1
                return self._formatOutputString(dest, 'Failed')
            job_ref = bq_processor_utils.ConstructObjectReference(job)
            self.truncated_table_count += 1
            return self._formatOutputString(dest, 'Successful %s ' % job_ref)
        except bq_error.BigqueryError as e:
            print(e)
            self.failed_table_count += 1
            return self._formatOutputString(dest, 'Failed %s ' % job_ref)