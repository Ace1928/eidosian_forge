from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _get_property(self):
    """Parse a property section."""
    line = self.next_line()
    if line is None:
        return None
    elif line.startswith(b'property '):
        return self._name_value(line[len(b'property '):])
    else:
        self.push_line(line)
        return None