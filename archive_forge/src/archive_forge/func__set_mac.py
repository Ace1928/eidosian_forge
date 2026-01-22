import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def _set_mac(self, network_id, dpid, port_no, port, mac_address):
    if not (port.network_id is None or port.network_id == network_id or port.network_id == self.nw_id_unknown):
        raise PortNotFound(network_id=network_id, dpid=dpid, port=port_no)
    port.network_id = network_id
    port.mac_address = mac_address
    if port.network_id and port.mac_address:
        self.send_event(EventMacAddress(dpid, port_no, port.network_id, port.mac_address, True))