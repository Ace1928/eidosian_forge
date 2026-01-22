import time
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver
from libcloud.common.xmlrpc import XMLRPCResponse, XMLRPCConnection
from libcloud.compute.types import Provider, NodeState
def _vcl_request(self, method, *args):
    res = self.connection.request(method, *args).object
    if res['status'] == 'error':
        raise LibcloudError(res['errormsg'], driver=self)
    return res