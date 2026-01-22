from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _decodeifneeded(self, value):
    if isinstance(value, bytes):
        if self.encoding:
            return value.decode(self.encoding)
        return value.decode()
    return value