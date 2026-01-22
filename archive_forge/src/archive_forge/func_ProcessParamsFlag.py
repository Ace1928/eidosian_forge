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
def ProcessParamsFlag(self, params, items):
    """Processes the params flag.

    Args:
      params: The user specified parameters. The parameters should be in JSON
        format given as a string. Ex: --params="{'param':'value'}".
      items: The body that contains information of all the flags set.

    Returns:
      items: The body after it has been updated with the params flag.

    Raises:
      bq_error.BigqueryError: If there is an error with the given params.
    """
    return bq_processor_utils.ProcessParamsFlag(params, items)