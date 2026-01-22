import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _handle_before_parameter_build(self, model, context, **kwargs):
    context['current_api_call_event'] = APICallEvent(service=model.service_model.service_id, operation=model.wire_name, timestamp=self._get_current_time())