import json
import base64
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeDriver, NodeLocation
from libcloud.compute.types import Provider
from libcloud.common.upcloud import (
class UpcloudConnection(ConnectionUserAndKey):
    """
    Connection class for UpcloudDriver
    """
    host = 'api.upcloud.com'
    responseCls = UpcloudResponse

    def add_default_headers(self, headers):
        """Adds headers that are needed for all requests"""
        headers['Authorization'] = self._basic_auth()
        headers['Accept'] = 'application/json'
        headers['Content-Type'] = 'application/json'
        return headers

    def _basic_auth(self):
        """Constructs basic auth header content string"""
        credentials = b('{}:{}'.format(self.user_id, self.key))
        credentials = base64.b64encode(credentials)
        return 'Basic {}'.format(credentials.decode('ascii'))