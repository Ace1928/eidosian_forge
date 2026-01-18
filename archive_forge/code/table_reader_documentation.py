from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
Read one page of data, up to max_rows rows.

    Assumes that the table is ready for reading. Will signal an error otherwise.

    Args:
      start_row: first row to read.
      max_rows: maximum number of rows to return.
      page_token: Optional. current page token.
      selected_fields: a subset of field to return.

    Returns:
      tuple of:
      rows: the actual rows of the table, in f,v format.
      page_token: the page token of the next page of results.
      schema: the schema of the table.
    