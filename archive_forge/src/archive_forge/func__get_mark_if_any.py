from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _get_mark_if_any(self):
    """Parse a mark section."""
    line = self.next_line()
    if line.startswith(b'mark :'):
        return line[len(b'mark :'):]
    else:
        self.push_line(line)
        return None