import sys
import hmac
import time
import uuid
import base64
import hashlib
from libcloud.utils.py3 import ET, b, u, urlquote
from libcloud.utils.xml import findtext
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import MalformedResponseError
def _sign_request(self, params, method, path):
    """
        Sign Aliyun requests parameters and get the signature.

        StringToSign = HTTPMethod + '&' +
                       percentEncode('/') + '&' +
                       percentEncode(CanonicalizedQueryString)
        """
    keys = list(params.keys())
    keys.sort()
    pairs = []
    for key in keys:
        pairs.append('{}={}'.format(_percent_encode(key), _percent_encode(params[key])))
    qs = urlquote('&'.join(pairs), safe='-_.~')
    string_to_sign = '&'.join((method, urlquote(path, safe=''), qs))
    b64_hmac = base64.b64encode(hmac.new(b(self._get_access_secret()), b(string_to_sign), digestmod=hashlib.sha1).digest())
    return b64_hmac.decode('utf8')