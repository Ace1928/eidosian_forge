from ncclient.xml_ import *
from ncclient.operations.rpc import RPC
class RebootMachine(RPC):
    """*reboot-machine* RPC (flowmon)"""
    DEPENDS = ['urn:liberouter:params:netconf:capability:power-control:1.0']

    def request(self):
        return self._request(new_ele(qualify('reboot-machine', PC_URN)))