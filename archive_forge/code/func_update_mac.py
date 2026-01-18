import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def update_mac(self, network_id, dpid, port_no, mac_address):
    old_mac_address = self._get_old_mac(network_id, dpid, port_no)
    self.dpids.update_mac(network_id, dpid, port_no, mac_address)
    if old_mac_address is not None:
        self.mac_addresses.remove_port(network_id, dpid, port_no, old_mac_address)
    self.mac_addresses.add_port(network_id, dpid, port_no, mac_address)