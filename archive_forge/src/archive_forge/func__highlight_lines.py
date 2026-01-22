from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _highlight_lines(self, tokensource):
    """
        Highlighted the lines specified in the `hl_lines` option by
        post-processing the token stream coming from `_format_lines`.
        """
    hls = self.hl_lines
    for i, (t, value) in enumerate(tokensource):
        if t != 1:
            yield (t, value)
        if i + 1 in hls:
            if self.noclasses:
                style = ''
                if self.style.highlight_color is not None:
                    style = ' style="background-color: %s"' % (self.style.highlight_color,)
                yield (1, '<span%s>%s</span>' % (style, value))
            else:
                yield (1, '<span class="hll">%s</span>' % value)
        else:
            yield (1, value)