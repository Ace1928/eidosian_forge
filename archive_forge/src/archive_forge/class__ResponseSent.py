import binascii
from .settings import ChangedSetting, _setting_code_from_int
class _ResponseSent(_HeadersSent):
    """
    The _ResponseSent event is fired whenever response headers are sent
    on a stream.

    This is an internal event, used to determine validation steps on
    outgoing header blocks.
    """
    pass