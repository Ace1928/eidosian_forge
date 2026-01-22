from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
def ReadSchemaAndRows(self, start_row: Optional[int], max_rows: Optional[int], selected_fields=None):
    """Read at most max_rows rows from a table and the schema.

    Args:
      start_row: first row to read.
      max_rows: maximum number of rows to return.
      selected_fields: a subset of fields to return.

    Raises:
      BigqueryInterfaceError: when bigquery returns something unexpected.
      ValueError: when start_row is None.
      ValueError: when max_rows is None.

    Returns:
      A tuple where the first item is the list of fields and the
      second item a list of rows.
    """
    if start_row is None:
        raise ValueError('start_row is required')
    if max_rows is None:
        raise ValueError('max_rows is required')
    page_token = None
    rows = []
    schema = {}
    while len(rows) < max_rows:
        rows_to_read = max_rows - len(rows)
        if not hasattr(self, 'max_rows_per_request'):
            raise NotImplementedError('Subclass must have max_rows_per_request instance variable')
        if self.max_rows_per_request:
            rows_to_read = min(self.max_rows_per_request, rows_to_read)
        more_rows, page_token, current_schema = self._ReadOnePage(None if page_token else start_row, max_rows=rows_to_read, page_token=page_token, selected_fields=selected_fields)
        if not schema and current_schema:
            schema = current_schema.get('fields', [])
        for row in more_rows:
            rows.append(self._ConvertFromFV(schema, row))
            start_row += 1
        if not page_token or not more_rows:
            break
    return (schema, rows)