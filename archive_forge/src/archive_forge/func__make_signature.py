import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
def _make_signature(self, params):
    signature = [(k.lower(), v) for k, v in list(params.items())]
    signature.sort(key=lambda x: x[0])
    pairs = []
    for pair in signature:
        key = urlquote(str(pair[0]), safe='[]')
        value = urlquote(str(pair[1]), safe='[]*')
        item = '{}={}'.format(key, value)
        pairs.append(item)
    signature = '&'.join(pairs)
    signature = signature.lower().replace('+', '%20')
    signature = hmac.new(b(self.key), msg=b(signature), digestmod=hashlib.sha1)
    return base64.b64encode(b(signature.digest()))