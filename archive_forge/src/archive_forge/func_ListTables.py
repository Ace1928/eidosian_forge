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
def ListTables(self, reference, max_results=None, page_token=None):
    """List the tables associated with this reference."""
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.DatasetReference, method='ListTables')
    request = bq_processor_utils.PrepareListRequest(reference, max_results, page_token)
    result = self.apiclient.tables().list(**request).execute()
    results = result.get('tables', [])
    if max_results is not None:
        while 'nextPageToken' in result and len(results) < max_results:
            request['maxResults'] = max_results - len(results)
            request['pageToken'] = result['nextPageToken']
            result = self.apiclient.tables().list(**request).execute()
            results.extend(result.get('tables', []))
    return results