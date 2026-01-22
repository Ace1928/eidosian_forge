import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _get_base_event_dict(self, event):
    return {'Version': 1, 'ClientId': self.csm_client_id}