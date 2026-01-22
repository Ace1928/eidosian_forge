import binascii
from .settings import ChangedSetting, _setting_code_from_int
class _RequestSent(_HeadersSent):
    """
    The _RequestSent event is fired whenever request headers are sent
    on a stream.

    This is an internal event, used to determine validation steps on
    outgoing header blocks.
    """
    pass