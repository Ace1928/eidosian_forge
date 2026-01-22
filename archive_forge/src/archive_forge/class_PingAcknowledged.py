import binascii
from .settings import ChangedSetting, _setting_code_from_int
class PingAcknowledged(Event):
    """
    Same as PingAckReceived.

    .. deprecated:: 3.1.0
    """

    def __init__(self):
        self.ping_data = None

    def __repr__(self):
        return '<PingAckReceived ping_data:%s>' % (_bytes_representation(self.ping_data),)