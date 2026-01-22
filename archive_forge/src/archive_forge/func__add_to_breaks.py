import fnmatch
import sys
import os
from inspect import CO_GENERATOR, CO_COROUTINE, CO_ASYNC_GENERATOR
def _add_to_breaks(self, filename, lineno):
    """Add breakpoint to breaks, if not already there."""
    bp_linenos = self.breaks.setdefault(filename, [])
    if lineno not in bp_linenos:
        bp_linenos.append(lineno)