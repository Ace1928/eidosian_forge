import logging
from os_ken.controller import event
from os_ken.lib.dpid import dpid_to_str
from os_ken.base import app_manager
class EventConfSwitchDelDPID(event.EventBase):

    def __init__(self, dpid):
        super(EventConfSwitchDelDPID, self).__init__()
        self.dpid = dpid

    def __str__(self):
        return 'EventConfSwitchDelDPID<%s>' % dpid_to_str(self.dpid)