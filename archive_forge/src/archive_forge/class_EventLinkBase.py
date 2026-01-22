import logging
from os_ken.controller import handler
from os_ken.controller import event
class EventLinkBase(event.EventBase):

    def __init__(self, link):
        super(EventLinkBase, self).__init__()
        self.link = link

    def __str__(self):
        return '%s<%s>' % (self.__class__.__name__, self.link)