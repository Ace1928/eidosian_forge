from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
class TableTableReader(_TableReader):
    """A TableReader that reads from a table."""

    def __init__(self, local_apiclient, max_rows_per_request, table_ref):
        self.table_ref = table_ref
        self.max_rows_per_request = max_rows_per_request
        self._apiclient = local_apiclient

    def _GetPrintContext(self) -> str:
        return '%r' % (self.table_ref,)

    def _ReadOnePage(self, start_row: Optional[int], max_rows: Optional[int], page_token: Optional[str]=None, selected_fields=None):
        kwds = dict(self.table_ref)
        kwds['maxResults'] = max_rows
        if page_token:
            kwds['pageToken'] = page_token
        else:
            kwds['startIndex'] = start_row
        data = None
        if selected_fields is not None:
            kwds['selectedFields'] = selected_fields
        if data is None:
            data = self._apiclient.tabledata().list(**kwds).execute()
        page_token = data.get('pageToken', None)
        rows = data.get('rows', [])
        kwds = dict(self.table_ref)
        if selected_fields is not None:
            kwds['selectedFields'] = selected_fields
        table_info = self._apiclient.tables().get(**kwds).execute()
        schema = table_info.get('schema', {})
        return (rows, page_token, schema)