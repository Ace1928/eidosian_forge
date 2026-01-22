from eventlet.event import Event
from eventlet import greenthread
import collections
class PropagateError(Exception):
    """
    When a DAGPool greenthread terminates with an exception instead of
    returning a result, attempting to retrieve its value raises
    PropagateError.

    Attributes:

    key
        the key of the greenthread which raised the exception

    exc
        the exception object raised by the greenthread
    """

    def __init__(self, key, exc):
        msg = 'PropagateError({}): {}: {}'.format(key, exc.__class__.__name__, exc)
        super().__init__(msg)
        self.msg = msg
        self.args = (key, exc)
        self.key = key
        self.exc = exc

    def __str__(self):
        return self.msg