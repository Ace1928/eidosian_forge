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
def InsertTableRows(self, table_dict: bq_id_utils.ApiClientHelper.TableReference, inserts: List[Optional[bq_processor_utils.InsertEntry]], skip_invalid_rows: Optional[bool]=None, ignore_unknown_values: Optional[bool]=None, template_suffix: Optional[int]=None):
    """Insert rows into a table.

    Arguments:
      table_dict: table reference into which rows are to be inserted.
      inserts: array of InsertEntry tuples where insert_id can be None.
      skip_invalid_rows: Optional. Attempt to insert any valid rows, even if
        invalid rows are present.
      ignore_unknown_values: Optional. Ignore any values in a row that are not
        present in the schema.
      template_suffix: Optional. The suffix used to generate the template
        table's name.

    Returns:
      result of the operation.
    """

    def _EncodeInsert(insert):
        encoded = dict(json=insert.record)
        if insert.insert_id:
            encoded['insertId'] = insert.insert_id
        return encoded
    op = self.GetInsertApiClient().tabledata().insertAll(body=dict(skipInvalidRows=skip_invalid_rows, ignoreUnknownValues=ignore_unknown_values, templateSuffix=template_suffix, rows=list(map(_EncodeInsert, inserts))), **table_dict)
    return op.execute()