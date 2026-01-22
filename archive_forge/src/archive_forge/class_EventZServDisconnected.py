from os_ken.controller.event import EventBase
class EventZServDisconnected(EventZClientBase):
    """
    The event class for notifying the disconnection from Zebra server.
    """

    def __init__(self, zserv):
        super(EventZServDisconnected, self).__init__()
        self.zserv = zserv