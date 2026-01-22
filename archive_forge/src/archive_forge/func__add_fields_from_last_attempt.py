import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _add_fields_from_last_attempt(self, event_dict, last_attempt):
    if last_attempt.request_headers:
        region = self._get_region(last_attempt.request_headers)
        if region is not None:
            event_dict['Region'] = region
        event_dict['UserAgent'] = self._get_user_agent(last_attempt.request_headers)
    if last_attempt.http_status_code is not None:
        event_dict['FinalHttpStatusCode'] = last_attempt.http_status_code
    if last_attempt.parsed_error is not None:
        self._serialize_parsed_error(last_attempt.parsed_error, event_dict, 'ApiCall')
    if last_attempt.wire_exception is not None:
        self._serialize_wire_exception(last_attempt.wire_exception, event_dict, 'ApiCall')