from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
class RimuHostingConnection(ConnectionKey):
    """
    Connection class for the RimuHosting driver
    """
    api_context = API_CONTEXT
    host = API_HOST
    port = 443
    responseCls = RimuHostingResponse

    def __init__(self, key, secure=True, retry_delay=None, backoff=None, timeout=None):
        ConnectionKey.__init__(self, key, secure, timeout=timeout, retry_delay=retry_delay, backoff=backoff)

    def add_default_headers(self, headers):
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'rimuhosting apikey=%s' % self.key
        return headers

    def request(self, action, params=None, data='', headers=None, method='GET'):
        if not headers:
            headers = {}
        if not params:
            params = {}
        return ConnectionKey.request(self, self.api_context + action, params, data, headers, method)