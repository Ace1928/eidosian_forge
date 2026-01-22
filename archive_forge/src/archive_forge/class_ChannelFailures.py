import atexit
import operator
import os
import sys
import threading
import time
import traceback as _traceback
import warnings
import subprocess
import functools
from more_itertools import always_iterable
class ChannelFailures(Exception):
    """Exception raised during errors on Bus.publish()."""
    delimiter = '\n'

    def __init__(self, *args, **kwargs):
        """Initialize ChannelFailures errors wrapper."""
        super(ChannelFailures, self).__init__(*args, **kwargs)
        self._exceptions = list()

    def handle_exception(self):
        """Append the current exception to self."""
        self._exceptions.append(sys.exc_info()[1])

    def get_instances(self):
        """Return a list of seen exception instances."""
        return self._exceptions[:]

    def __str__(self):
        """Render the list of errors, which happened in channel."""
        exception_strings = map(repr, self.get_instances())
        return self.delimiter.join(exception_strings)
    __repr__ = __str__

    def __bool__(self):
        """Determine whether any error happened in channel."""
        return bool(self._exceptions)
    __nonzero__ = __bool__