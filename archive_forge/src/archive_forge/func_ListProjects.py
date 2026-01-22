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
def ListProjects(self, max_results=None, page_token=None):
    """List the projects this user has access to."""
    request = bq_processor_utils.PrepareListRequest({}, max_results, page_token)
    result = self._ExecuteListProjectsRequest(request)
    results = result.get('projects', [])
    while 'nextPageToken' in result and (max_results is not None and len(results) < max_results):
        request['pageToken'] = result['nextPageToken']
        result = self._ExecuteListProjectsRequest(request)
        results.extend(result.get('projects', []))
    return results