import functools
import os
import sys
import os.path
from io import StringIO
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt
def _get_css_inline_styles(self, ttype):
    """Return the inline CSS styles for this token type."""
    cclass = self.ttype2class.get(ttype)
    while cclass is None:
        ttype = ttype.parent
        cclass = self.ttype2class.get(ttype)
    return cclass or ''