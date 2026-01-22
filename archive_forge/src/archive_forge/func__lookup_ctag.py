from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _lookup_ctag(self, token):
    entry = ctags.TagEntry()
    if self._ctags.find(entry, token, 0):
        return (entry['file'], entry['lineNumber'])
    else:
        return (None, None)