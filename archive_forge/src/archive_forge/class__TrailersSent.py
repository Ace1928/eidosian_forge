import binascii
from .settings import ChangedSetting, _setting_code_from_int
class _TrailersSent(_HeadersSent):
    """
    The _TrailersSent event is fired whenever trailers are sent on a
    stream. Trailers are a set of headers sent after the body of the
    request/response, and are used to provide information that wasn't known
    ahead of time (e.g. content-length).

    This is an internal event, used to determine validation steps on
    outgoing header blocks.
    """
    pass