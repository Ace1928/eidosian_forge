from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Optional
from utils import bq_error
class JobTableReader(_TableReader):
    """A TableReader that reads from a completed job."""

    def __init__(self, local_apiclient, max_rows_per_request, job_ref):
        self.job_ref = job_ref
        self.max_rows_per_request = max_rows_per_request
        self._apiclient = local_apiclient

    def _GetPrintContext(self) -> str:
        return '%r' % (self.job_ref,)

    def _ReadOnePage(self, start_row: Optional[int], max_rows: Optional[int], page_token: Optional[str]=None, selected_fields=None):
        kwds = dict(self.job_ref)
        kwds['maxResults'] = max_rows
        kwds['timeoutMs'] = 0
        if page_token:
            kwds['pageToken'] = page_token
        else:
            kwds['startIndex'] = start_row
        data = self._apiclient.jobs().getQueryResults(**kwds).execute()
        if not data['jobComplete']:
            raise bq_error.BigqueryError('Job %s is not done' % (self,))
        page_token = data.get('pageToken', None)
        schema = data.get('schema', None)
        rows = data.get('rows', [])
        return (rows, page_token, schema)