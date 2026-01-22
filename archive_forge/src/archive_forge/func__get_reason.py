from __future__ import absolute_import
import json
import six
from googleapiclient import _helpers as util
def _get_reason(self):
    """Calculate the reason for the error from the response content."""
    reason = self.resp.reason if hasattr(self.resp, 'reason') else None
    try:
        try:
            data = json.loads(self.content.decode('utf-8'))
        except json.JSONDecodeError:
            data = self.content.decode('utf-8')
        if isinstance(data, dict):
            reason = data['error']['message']
            error_detail_keyword = next((kw for kw in ['detail', 'details', 'errors', 'message'] if kw in data['error']), '')
            if error_detail_keyword:
                self.error_details = data['error'][error_detail_keyword]
        elif isinstance(data, list) and len(data) > 0:
            first_error = data[0]
            reason = first_error['error']['message']
            if 'details' in first_error['error']:
                self.error_details = first_error['error']['details']
        else:
            self.error_details = data
    except (ValueError, KeyError, TypeError):
        pass
    if reason is None:
        reason = ''
    return reason