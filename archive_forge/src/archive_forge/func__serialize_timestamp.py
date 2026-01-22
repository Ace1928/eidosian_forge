import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _serialize_timestamp(self, timestamp, event_dict, **kwargs):
    event_dict['Timestamp'] = timestamp