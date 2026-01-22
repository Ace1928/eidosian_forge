import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class LiquidWebConnection(ConnectionUserAndKey):
    host = API_HOST
    responseCls = LiquidWebResponse

    def add_default_headers(self, headers):
        b64string = b('{}:{}'.format(self.user_id, self.key))
        encoded = base64.b64encode(b64string).decode('utf-8')
        authorization = 'Basic ' + encoded
        headers['Authorization'] = authorization
        headers['Content-Type'] = 'application/json'
        return headers