import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _handle_after_call(self, context, parsed, **kwargs):
    context['current_api_call_event'].retries_exceeded = parsed['ResponseMetadata'].get('MaxAttemptsReached', False)
    return self._complete_api_call(context)