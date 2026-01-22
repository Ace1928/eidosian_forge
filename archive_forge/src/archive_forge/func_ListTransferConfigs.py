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
def ListTransferConfigs(self, reference=None, location=None, page_size=None, page_token=None, data_source_ids=None):
    """Return a list of transfer configurations.

    Args:
      reference: The ProjectReference to list transfer configurations for.
      location: The location id, e.g. 'us' or 'eu'.
      page_size: The maximum number of transfer configurations to return.
      page_token: Current page token (optional).
      data_source_ids: The dataSourceIds to display transfer configurations for.

    Returns:
      A list of transfer configurations.
    """
    results = None
    client = self.GetTransferV1ApiClient()
    bq_id_utils.typecheck(reference, bq_id_utils.ApiClientHelper.ProjectReference, method='ListTransferConfigs')
    if page_size is not None:
        if page_size > bq_processor_utils.MAX_RESULTS:
            page_size = bq_processor_utils.MAX_RESULTS
    request = bq_processor_utils.PrepareTransferListRequest(reference, location, page_size, page_token, data_source_ids)
    if request:
        bq_processor_utils.ApplyParameters(request)
        result = client.projects().locations().transferConfigs().list(**request).execute()
        results = result.get('transferConfigs', [])
        if page_size is not None:
            while 'nextPageToken' in result and len(results) < page_size:
                request = bq_processor_utils.PrepareTransferListRequest(reference, location, page_size - len(results), result['nextPageToken'], data_source_ids)
                if request:
                    bq_processor_utils.ApplyParameters(request)
                    result = client.projects().locations().transferConfigs().list(**request).execute()
                    results.extend(result.get('nextPageToken', []))
                else:
                    return
        if len(results) < 1:
            logging.info('There are no transfer configurations to be shown.')
        if result.get('nextPageToken'):
            return (results, result.get('nextPageToken'))
    return (results,)