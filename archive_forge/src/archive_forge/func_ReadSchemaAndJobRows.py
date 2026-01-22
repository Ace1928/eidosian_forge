from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional
import uuid
from absl import flags
from google.api_core.iam import Policy
from googleapiclient import http as http_request
import inflection
from clients import bigquery_client
from clients import client_dataset
from clients import client_reservation
from clients import table_reader as bq_table_reader
from clients import utils as bq_client_utils
from utils import bq_api_utils
from utils import bq_error
from utils import bq_id_utils
from utils import bq_processor_utils
def ReadSchemaAndJobRows(self, job_dict, start_row=None, max_rows=None, result_first_page=None):
    """Convenience method to get the schema and rows from job query result.

    Arguments:
      job_dict: job reference dictionary.
      start_row: first row to read.
      max_rows: number of rows to read.
      result_first_page: the first page of the result of a query job.

    Returns:
      A tuple where the first item is the list of fields and the
      second item a list of rows.
    Raises:
      ValueError: will be raised if start_row is not explicitly provided.
      ValueError: will be raised if max_rows is not explicitly provided.
    """
    if start_row is None:
        raise ValueError('start_row is required')
    if max_rows is None:
        raise ValueError('max_rows is required')
    if not job_dict:
        job_ref: bq_id_utils.ApiClientHelper.JobReference = None
    else:
        job_ref = bq_id_utils.ApiClientHelper.JobReference.Create(**job_dict)
    if flags.FLAGS.jobs_query_use_results_from_response and result_first_page:
        reader = bq_table_reader.QueryTableReader(self.apiclient, self.max_rows_per_request, job_ref, result_first_page)
    else:
        reader = bq_table_reader.JobTableReader(self.apiclient, self.max_rows_per_request, job_ref)
    return reader.ReadSchemaAndRows(start_row, max_rows)