import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchRequest(event.EventRequestBase):

    def __init__(self, dpid=None):
        super(EventSwitchRequest, self).__init__()
        self.dst = 'switches'
        self.dpid = dpid

    def __str__(self):
        return 'EventSwitchRequest<src=%s, dpid=%s>' % (self.src, self.dpid)