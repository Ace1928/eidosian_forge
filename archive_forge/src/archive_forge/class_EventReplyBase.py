class EventReplyBase(EventBase):
    """
    The base class for synchronous request reply for OSKenApp.send_reply.
    """

    def __init__(self, dst):
        super(EventReplyBase, self).__init__()
        self.dst = dst