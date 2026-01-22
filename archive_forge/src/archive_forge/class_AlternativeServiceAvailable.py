import binascii
from .settings import ChangedSetting, _setting_code_from_int
class AlternativeServiceAvailable(Event):
    """
    The AlternativeServiceAvailable event is fired when the remote peer
    advertises an `RFC 7838 <https://tools.ietf.org/html/rfc7838>`_ Alternative
    Service using an ALTSVC frame.

    This event always carries the origin to which the ALTSVC information
    applies. That origin is either supplied by the server directly, or inferred
    by hyper-h2 from the ``:authority`` pseudo-header field that was sent by
    the user when initiating a given stream.

    This event also carries what RFC 7838 calls the "Alternative Service Field
    Value", which is formatted like a HTTP header field and contains the
    relevant alternative service information. Hyper-h2 does not parse or in any
    way modify that information: the user is required to do that.

    This event can only be fired on the client end of a connection.

    .. versionadded:: 2.3.0
    """

    def __init__(self):
        self.origin = None
        self.field_value = None

    def __repr__(self):
        return '<AlternativeServiceAvailable origin:%s, field_value:%s>' % (self.origin.decode('utf-8', 'ignore'), self.field_value.decode('utf-8', 'ignore'))