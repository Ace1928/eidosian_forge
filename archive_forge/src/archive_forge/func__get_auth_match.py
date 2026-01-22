import json
import logging
import re
import time
from botocore.compat import ensure_bytes, ensure_unicode, urlparse
from botocore.retryhandler import EXCEPTION_MAP as RETRYABLE_EXCEPTIONS
def _get_auth_match(self, auth_val):
    for signature_version, regex in self._AUTH_REGEXS.items():
        match = regex.match(auth_val)
        if match:
            return (signature_version, match)
    return (None, None)