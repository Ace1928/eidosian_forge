import os
import sys
import linecache
import re
import inspect
def tpool_exceptions(state=False):
    """Toggles whether tpool itself prints exceptions that are raised from
    functions that are executed in it, in addition to raising them like
    it normally does."""
    from eventlet import tpool
    tpool.QUIET = not state