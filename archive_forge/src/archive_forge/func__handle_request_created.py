import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _handle_request_created(self, request, **kwargs):
    context = request.context
    new_attempt_event = context['current_api_call_event'].new_api_call_attempt(timestamp=self._get_current_time())
    new_attempt_event.request_headers = request.headers
    new_attempt_event.url = request.url
    context['current_api_call_attempt_event'] = new_attempt_event