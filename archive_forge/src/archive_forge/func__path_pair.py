from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _path_pair(self, s):
    """Parse two paths separated by a space."""
    if s.startswith(b'"'):
        parts = s[1:].split(b'" ', 1)
    else:
        parts = s.split(b' ', 1)
    if len(parts) != 2:
        self.abort(errors.BadFormat, '?', '?', s)
    elif parts[1].startswith(b'"') and parts[1].endswith(b'"'):
        parts[1] = parts[1][1:-1]
    elif parts[1].startswith(b'"') or parts[1].endswith(b'"'):
        self.abort(errors.BadFormat, '?', '?', s)
    return [_unquote_c_string(part) for part in parts]