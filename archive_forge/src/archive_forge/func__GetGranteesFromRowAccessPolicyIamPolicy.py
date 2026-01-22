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
def _GetGranteesFromRowAccessPolicyIamPolicy(self, iam_policy):
    """Returns the filtered data viewer members of the given IAM policy."""
    bindings = iam_policy.get('bindings')
    if not bindings:
        return []
    filtered_data_viewer_binding = next((binding for binding in bindings if binding.get('role') == _FILTERED_DATA_VIEWER_ROLE), None)
    if not filtered_data_viewer_binding:
        return []
    return filtered_data_viewer_binding.get('members', [])