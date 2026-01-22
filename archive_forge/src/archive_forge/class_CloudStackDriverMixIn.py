import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
class CloudStackDriverMixIn:
    host = None
    path = None
    connectionCls = CloudStackConnection

    def __init__(self, key, secret=None, secure=True, host=None, port=None):
        host = host or self.host
        super().__init__(key, secret, secure, host, port)

    def _sync_request(self, command, action=None, params=None, data=None, headers=None, method='GET'):
        return self.connection._sync_request(command=command, action=action, params=params, data=data, headers=headers, method=method)

    def _async_request(self, command, action=None, params=None, data=None, headers=None, method='GET', context=None):
        return self.connection._async_request(command=command, action=action, params=params, data=data, headers=headers, method=method, context=context)