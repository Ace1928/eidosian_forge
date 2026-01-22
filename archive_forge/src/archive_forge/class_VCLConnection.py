import time
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
from libcloud.compute.types import Provider, NodeState
class VCLConnection(XMLRPCConnection, ConnectionUserAndKey):
    endpoint = '/index.php?mode=xmlrpccall'

    def add_default_headers(self, headers):
        headers['X-APIVERSION'] = '2'
        headers['X-User'] = self.user_id
        headers['X-Pass'] = self.key
        return headers