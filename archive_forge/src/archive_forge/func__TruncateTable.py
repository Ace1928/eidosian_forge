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