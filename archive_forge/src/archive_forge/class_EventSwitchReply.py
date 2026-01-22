import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventSwitchReply(event.EventReplyBase):

    def __init__(self, dst, switches):
        super(EventSwitchReply, self).__init__(dst)
        self.switches = switches

    def __str__(self):
        return 'EventSwitchReply<dst=%s, %s>' % (self.dst, self.switches)