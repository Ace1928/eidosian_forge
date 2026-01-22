import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_wire_exception(self, wire_exception, event_dict, event_type, **kwargs):
    field_prefix = 'Final' if event_type == 'ApiCall' else ''
    event_dict[field_prefix + 'SdkException'] = self._truncate(wire_exception.__class__.__name__, self._MAX_EXCEPTION_CLASS_LENGTH)
    event_dict[field_prefix + 'SdkExceptionMessage'] = self._truncate(str(wire_exception), self._MAX_MESSAGE_LENGTH)