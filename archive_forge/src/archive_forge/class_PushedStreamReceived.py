import binascii
from .settings import ChangedSetting, _setting_code_from_int
class PushedStreamReceived(Event):
    """
    The PushedStreamReceived event is fired whenever a pushed stream has been
    received from a remote peer. The event carries on it the new stream ID, the
    ID of the parent stream, and the request headers pushed by the remote peer.
    """

    def __init__(self):
        self.pushed_stream_id = None
        self.parent_stream_id = None
        self.headers = None

    def __repr__(self):
        return '<PushedStreamReceived pushed_stream_id:%s, parent_stream_id:%s, headers:%s>' % (self.pushed_stream_id, self.parent_stream_id, self.headers)