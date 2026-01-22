import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def _calc_signature_1(self, params):
    """Generate AWS signature version 1 string."""
    for key in sorted(params, key=str.lower):
        self.hmac.update(key.encode('utf-8'))
        val = self._get_utf8_value(params[key])
        self.hmac.update(val)
    return base64.b64encode(self.hmac.digest()).decode('utf-8')