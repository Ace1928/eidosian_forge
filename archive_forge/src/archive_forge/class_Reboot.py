from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
from ncclient.operations.rpc import RPCReply
from ncclient.operations.rpc import RPCError
from ncclient import NCClientError
import math
class Reboot(RPC):

    def request(self):
        node = new_ele('request-reboot')
        return self._request(node)