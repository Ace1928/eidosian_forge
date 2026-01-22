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
@staticmethod
def ConfigureFormatter(formatter, reference_type, print_format='list', object_info=None):
    """Configure a formatter for a given reference type.

    If print_format is 'show', configures the formatter with several
    additional fields (useful for printing a single record).

    Arguments:
      formatter: TableFormatter object to configure.
      reference_type: Type of object this formatter will be used with.
      print_format: Either 'show' or 'list' to control what fields are included.

    Raises:
      ValueError: If reference_type or format is unknown.
    """
    return bq_client_utils.ConfigureFormatter(formatter, reference_type, print_format, object_info)