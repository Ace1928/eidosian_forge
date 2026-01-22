from os_ken.controller.event import EventBase
class EventZServConnected(EventZClientBase):
    """
    The event class for notifying the connection to Zebra server.
    """

    def __init__(self, zserv):
        super(EventZServConnected, self).__init__()
        self.zserv = zserv