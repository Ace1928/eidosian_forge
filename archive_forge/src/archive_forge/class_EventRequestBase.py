class EventRequestBase(EventBase):
    """
    The base class for synchronous request for OSKenApp.send_request.
    """

    def __init__(self):
        super(EventRequestBase, self).__init__()
        self.dst = None
        self.src = None
        self.sync = False
        self.reply_q = None