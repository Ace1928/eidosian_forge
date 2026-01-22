from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations.rpc import RPCReply
from ncclient.operations.rpc import RPCError
from ncclient import NCClientError
import math
class GetConfiguration(RPC):

    def request(self, format='xml', filter=None):
        node = new_ele('get-configuration', {'format': format})
        if filter is not None:
            node.append(filter)
        if format != 'xml':
            self._huge_tree = True
        return self._request(node)