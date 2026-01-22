from suds import *
from logging import getLogger
class MessageContext(Context):
    """
    The context for sending the SOAP envelope.

    @ivar envelope: The SOAP envelope to be sent.
    @type envelope: (str|L{sax.element.Element})
    @ivar reply: The reply.
    @type reply: (str|L{sax.element.Element}|object)

    """
    pass