import weakref
from pydispatch import saferef, robustapply, errors
class _Anonymous(_Parameter):
    """Singleton used to signal "Anonymous Sender"

    The Anonymous object is used to signal that the sender
    of a message is not specified (as distinct from being
    "any sender").  Registering callbacks for Anonymous
    will only receive messages sent without senders.  Sending
    with anonymous will only send messages to those receivers
    registered for Any or Anonymous.

    Note:
        The default sender for connect is Any, while the
        default sender for send is Anonymous.  This has
        the effect that if you do not specify any senders
        in either function then all messages are routed
        as though there was a single sender (Anonymous)
        being used everywhere.
    """