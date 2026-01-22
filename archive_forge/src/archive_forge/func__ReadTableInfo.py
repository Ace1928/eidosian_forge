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