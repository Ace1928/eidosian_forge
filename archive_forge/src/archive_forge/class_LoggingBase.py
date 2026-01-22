from collections import namedtuple
from testtools.tags import TagContext
class LoggingBase:
    """Basic support for logging of results."""

    def __init__(self, event_log=None):
        if event_log is None:
            event_log = []
        self._events = event_log