from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPListRequest(event.EventRequestBase):
    """
    Event that requests list of configured VRRP router
    instance_name=None means all instances.
    """

    def __init__(self, instance_name=None):
        super(EventVRRPListRequest, self).__init__()
        self.instance_name = instance_name