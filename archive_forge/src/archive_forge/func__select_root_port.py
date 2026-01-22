import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def _select_root_port(self):
    """ ROOT_PORT is the nearest port to a root bridge.
            It is determined by the cost of path, etc. """
    root_port = None
    for port in self.ports.values():
        root_msg = self.root_priority if root_port is None else root_port.designated_priority
        port_msg = port.designated_priority
        if port.state is PORT_STATE_DISABLE or port_msg is None:
            continue
        if root_msg.root_id.value > port_msg.root_id.value:
            result = SUPERIOR
        elif root_msg.root_id.value == port_msg.root_id.value:
            if root_msg.designated_bridge_id is None:
                result = INFERIOR
            else:
                result = Stp.compare_root_path(port_msg.root_path_cost, root_msg.root_path_cost, port_msg.designated_bridge_id.value, root_msg.designated_bridge_id.value, port_msg.designated_port_id.value, root_msg.designated_port_id.value)
        else:
            result = INFERIOR
        if result is SUPERIOR:
            root_port = port
    return root_port