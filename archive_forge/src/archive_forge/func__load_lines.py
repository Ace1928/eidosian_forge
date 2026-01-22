import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _load_lines(self):
    """Private API. force all lines in the stack to be loaded."""
    for frame in self.stack:
        frame.line