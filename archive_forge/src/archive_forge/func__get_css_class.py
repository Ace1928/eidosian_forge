from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _get_css_class(self, ttype):
    """Return the css class of this token type prefixed with
        the classprefix option."""
    ttypeclass = _get_ttype_class(ttype)
    if ttypeclass:
        return self.classprefix + ttypeclass
    return ''