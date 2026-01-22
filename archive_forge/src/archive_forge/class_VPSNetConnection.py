import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
class VPSNetConnection(ConnectionUserAndKey):
    """
    Connection class for the VPS.net driver
    """
    host = API_HOST
    responseCls = VPSNetResponse
    allow_insecure = False

    def add_default_headers(self, headers):
        user_b64 = base64.b64encode(b('{}:{}'.format(self.user_id, self.key)))
        headers['Authorization'] = 'Basic %s' % user_b64.decode('utf-8')
        return headers