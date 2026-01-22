import collections
from os_ken.base import app_manager
import os_ken.exception as os_ken_exc
from os_ken.controller import event
from os_ken.exception import NetworkNotFound, NetworkAlreadyExist
from os_ken.exception import PortAlreadyExist, PortNotFound, PortUnknown
def _update_port(self, network_id, dpid, port, port_may_exist):

    def _known_nw_id(nw_id):
        return nw_id is not None and nw_id != self.nw_id_unknown
    queue_add_event = False
    self._check_nw_id_unknown(network_id)
    try:
        old_network_id = self.dpids.get_network_safe(dpid, port)
        if self.networks.has_port(network_id, dpid, port) or _known_nw_id(old_network_id):
            if not port_may_exist:
                raise PortAlreadyExist(network_id=network_id, dpid=dpid, port=port)
        if old_network_id != network_id:
            queue_add_event = True
            self.networks.add_raw(network_id, dpid, port)
            if _known_nw_id(old_network_id):
                self.networks.remove_raw(old_network_id, dpid, port)
    except KeyError:
        raise NetworkNotFound(network_id=network_id)
    self.dpids.update_port(dpid, port, network_id)
    if queue_add_event:
        self.networks.add_event(network_id, dpid, port)