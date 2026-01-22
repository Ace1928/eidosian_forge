import binascii
from .settings import ChangedSetting, _setting_code_from_int
class ConnectionTerminated(Event):
    """
    The ConnectionTerminated event is fired when a connection is torn down by
    the remote peer using a GOAWAY frame. Once received, no further action may
    be taken on the connection: a new connection must be established.
    """

    def __init__(self):
        self.error_code = None
        self.last_stream_id = None
        self.additional_data = None

    def __repr__(self):
        return '<ConnectionTerminated error_code:%s, last_stream_id:%s, additional_data:%s>' % (self.error_code, self.last_stream_id, _bytes_representation(self.additional_data[:20] if self.additional_data else None))