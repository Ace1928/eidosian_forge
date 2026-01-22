from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import sys
from typing import Optional
from absl import app
from absl import flags
from pyglib import appcommands
from clients import bigquery_client
from clients import bigquery_client_extended
from clients import utils as bq_client_utils
from frontend import bigquery_command
from frontend import bq_cached_client
from frontend import utils as frontend_utils
from frontend import utils_data_transfer
from utils import bq_error
from utils import bq_id_utils
from pyglib import stringutil
def PrintNonScriptQueryJobResults(self, client: bigquery_client_extended.BigqueryClientExtended, job) -> None:
    printable_job_info = bq_client_utils.FormatJobInfo(job)
    is_assert_job = job['statistics']['query']['statementType'] == 'ASSERT'
    if not bq_client_utils.IsFailedJob(job) and (not frontend_utils.IsSuccessfulDmlOrDdlJob(printable_job_info)) and (not is_assert_job):
        fields, rows = client.ReadSchemaAndJobRows(job['jobReference'], start_row=self.start_row, max_rows=self.max_rows)
        bq_cached_client.Factory.ClientTablePrinter.GetTablePrinter().PrintTable(fields, rows)
    frontend_utils.PrintJobMessages(printable_job_info)