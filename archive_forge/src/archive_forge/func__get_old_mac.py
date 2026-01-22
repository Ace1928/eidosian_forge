import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def _get_old_mac(self, network_id, dpid, port_no):
    try:
        port = self.dpids.get_port(dpid, port_no)
    except PortNotFound:
        pass
    else:
        if port.network_id == network_id:
            return port.mac_address
    return None