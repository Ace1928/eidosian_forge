import binascii
from .settings import ChangedSetting, _setting_code_from_int
class _PushedRequestSent(_HeadersSent):
    """
    The _PushedRequestSent event is fired whenever pushed request headers are
    sent.

    This is an internal event, used to determine validation steps on outgoing
    header blocks.
    """
    pass