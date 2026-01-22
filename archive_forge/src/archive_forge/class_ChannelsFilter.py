import sys
import logging
import timeit
from functools import wraps
from collections.abc import Mapping, Callable
import warnings
from logging import PercentStyle
class ChannelsFilter(logging.Filter):
    """Provides a hierarchical filter for log entries based on channel names.

    Filters out records emitted from a list of enabled channel names,
    including their children. It works the same as the ``logging.Filter``
    class, but allows the user to specify multiple channel names.

    >>> import sys
    >>> handler = logging.StreamHandler(sys.stdout)
    >>> handler.setFormatter(logging.Formatter("%(message)s"))
    >>> filter = ChannelsFilter("A.B", "C.D")
    >>> handler.addFilter(filter)
    >>> root = logging.getLogger()
    >>> root.addHandler(handler)
    >>> root.setLevel(level=logging.DEBUG)
    >>> logging.getLogger('A.B').debug('this record passes through')
    this record passes through
    >>> logging.getLogger('A.B.C').debug('records from children also pass')
    records from children also pass
    >>> logging.getLogger('C.D').debug('this one as well')
    this one as well
    >>> logging.getLogger('A.B.').debug('also this one')
    also this one
    >>> logging.getLogger('A.F').debug('but this one does not!')
    >>> logging.getLogger('C.DE').debug('neither this one!')
    """

    def __init__(self, *names):
        self.names = names
        self.num = len(names)
        self.lengths = {n: len(n) for n in names}

    def filter(self, record):
        if self.num == 0:
            return True
        for name in self.names:
            nlen = self.lengths[name]
            if name == record.name:
                return True
            elif record.name.find(name, 0, nlen) == 0 and record.name[nlen] == '.':
                return True
        return False