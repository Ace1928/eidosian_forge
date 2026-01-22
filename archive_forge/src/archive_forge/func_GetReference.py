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
def GetReference(self, identifier=''):
    """Try to deduce a project/dataset/table reference from a string.

    If the identifier is not compound, treat it as the most specific
    identifier we don't have as a flag, or as the table_id. If it is
    compound, fill in any unspecified part.

    Args:
      identifier: string, Identifier to create a reference for.

    Returns:
      A valid ProjectReference, DatasetReference, or TableReference.

    Raises:
      bq_error.BigqueryError: if no valid reference can be determined.
    """
    return bq_client_utils.GetReference(id_fallbacks=self, identifier=identifier)