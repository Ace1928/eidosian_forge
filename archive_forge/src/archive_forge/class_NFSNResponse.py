import time
import random
import string
import hashlib
from libcloud.utils.py3 import httplib, urlencode, basestring
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
class NFSNResponse(JsonResponse):

    def parse_error(self):
        if self.status == httplib.UNAUTHORIZED:
            raise InvalidCredsError('Invalid provider credentials')
        body = self.parse_body()
        if isinstance(body, basestring):
            return body + ' (HTTP Code: %d)' % self.status
        error = body.get('error', None)
        debug = body.get('debug', None)
        value = 'No message specified'
        if error is not None:
            value = error
        if debug is not None:
            value = debug
        if error is not None and value is not None:
            value = error + ' ' + value
        value = value + ' (HTTP Code: %d)' % self.status
        return value