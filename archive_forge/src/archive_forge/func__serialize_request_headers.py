import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_request_headers(self, request_headers, event_dict, **kwargs):
    event_dict['UserAgent'] = self._get_user_agent(request_headers)
    if self._is_signed(request_headers):
        event_dict['AccessKey'] = self._get_access_key(request_headers)
    region = self._get_region(request_headers)
    if region is not None:
        event_dict['Region'] = region
    if 'X-Amz-Security-Token' in request_headers:
        event_dict['SessionToken'] = request_headers['X-Amz-Security-Token']