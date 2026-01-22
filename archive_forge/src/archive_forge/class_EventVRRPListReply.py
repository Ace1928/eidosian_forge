from os_ken.controller import handler
from os_ken.controller import event
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import vrrp
from os_ken.lib import addrconv
class EventVRRPListReply(event.EventReplyBase):

    def __init__(self, instance_list):
        super(EventVRRPListReply, self).__init__(None)
        self.instance_list = instance_list