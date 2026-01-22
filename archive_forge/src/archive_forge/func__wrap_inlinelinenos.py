from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_inlinelinenos(self, inner):
    lines = list(inner)
    sp = self.linenospecial
    st = self.linenostep
    num = self.linenostart
    mw = len(str(len(lines) + num - 1))
    if self.noclasses:
        if sp:
            for t, line in lines:
                if num % sp == 0:
                    style = 'background-color: #ffffc0; padding: 0 5px 0 5px'
                else:
                    style = 'background-color: #f0f0f0; padding: 0 5px 0 5px'
                yield (1, '<span style="%s">%*s </span>' % (style, mw, num % st and ' ' or num) + line)
                num += 1
        else:
            for t, line in lines:
                yield (1, '<span style="background-color: #f0f0f0; padding: 0 5px 0 5px">%*s </span>' % (mw, num % st and ' ' or num) + line)
                num += 1
    elif sp:
        for t, line in lines:
            yield (1, '<span class="lineno%s">%*s </span>' % (num % sp == 0 and ' special' or '', mw, num % st and ' ' or num) + line)
            num += 1
    else:
        for t, line in lines:
            yield (1, '<span class="lineno">%*s </span>' % (mw, num % st and ' ' or num) + line)
            num += 1