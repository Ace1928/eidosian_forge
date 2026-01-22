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
def _clean_exit(self):
    """Assert that the Bus is not running in atexit handler callback."""
    if self.state != states.EXITING:
        warnings.warn('The main thread is exiting, but the Bus is in the %r state; shutting it down automatically now. You must either call bus.block() after start(), or call bus.exit() before the main thread exits.' % self.state, RuntimeWarning)
        self.exit()