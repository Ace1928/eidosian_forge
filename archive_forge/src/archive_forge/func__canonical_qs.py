import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
@staticmethod
def _canonical_qs(params):
    """Construct a sorted, correctly encoded query string.

        This is required for _calc_signature_2 and _calc_signature_4.
        """
    keys = list(params)
    keys.sort()
    pairs = []
    for key in keys:
        val = Ec2Signer._get_utf8_value(params[key])
        val = urllib.parse.quote(val, safe='-_~')
        pairs.append(urllib.parse.quote(key, safe='') + '=' + val)
    qs = '&'.join(pairs)
    return qs