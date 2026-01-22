from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_div(self, inner):
    style = []
    if self.noclasses and (not self.nobackground) and (self.style.background_color is not None):
        style.append('background: %s' % (self.style.background_color,))
    if self.cssstyles:
        style.append(self.cssstyles)
    style = '; '.join(style)
    yield (0, '<div' + (self.cssclass and ' class="%s"' % self.cssclass) + (style and ' style="%s"' % style) + '>')
    for tup in inner:
        yield tup
    yield (0, '</div>\n')