from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
@handler.set_ev_handler(vrrp_event.EventVRRPTransmitRequest)
def vrrp_transmit_request_handler(self, ev):
    raise NotImplementedError()