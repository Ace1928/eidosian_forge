import weakref
from pydispatch import saferef, robustapply, errors
class _Any(_Parameter):
    """Singleton used to signal either "Any Sender" or "Any Signal"

    The Any object can be used with connect, disconnect,
    send, or sendExact to signal that the parameter given
    Any should react to all senders/signals, not just
    a particular sender/signal.
    """