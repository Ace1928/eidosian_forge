from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPReceived(event.EventBase):
    """
    Event that port manager received valid VRRP packet.
    Usually handed by VRRP Router.
    """

    def __init__(self, interface, packet):
        super(EventVRRPReceived, self).__init__()
        self.interface = interface
        self.packet = packet