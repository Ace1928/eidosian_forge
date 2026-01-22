from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def _wrap_tablelinenos(self, inner):
    dummyoutfile = StringIO()
    lncount = 0
    for t, line in inner:
        if t:
            lncount += 1
        dummyoutfile.write(line)
    fl = self.linenostart
    mw = len(str(lncount + fl - 1))
    sp = self.linenospecial
    st = self.linenostep
    la = self.lineanchors
    aln = self.anchorlinenos
    nocls = self.noclasses
    if sp:
        lines = []
        for i in range(fl, fl + lncount):
            if i % st == 0:
                if i % sp == 0:
                    if aln:
                        lines.append('<a href="#%s-%d" class="special">%*d</a>' % (la, i, mw, i))
                    else:
                        lines.append('<span class="special">%*d</span>' % (mw, i))
                elif aln:
                    lines.append('<a href="#%s-%d">%*d</a>' % (la, i, mw, i))
                else:
                    lines.append('%*d' % (mw, i))
            else:
                lines.append('')
        ls = '\n'.join(lines)
    else:
        lines = []
        for i in range(fl, fl + lncount):
            if i % st == 0:
                if aln:
                    lines.append('<a href="#%s-%d">%*d</a>' % (la, i, mw, i))
                else:
                    lines.append('%*d' % (mw, i))
            else:
                lines.append('')
        ls = '\n'.join(lines)
    if nocls:
        yield (0, '<table class="%stable">' % self.cssclass + '<tr><td><div class="linenodiv" style="background-color: #f0f0f0; padding-right: 10px"><pre style="line-height: 125%">' + ls + '</pre></div></td><td class="code">')
    else:
        yield (0, '<table class="%stable">' % self.cssclass + '<tr><td class="linenos"><div class="linenodiv"><pre>' + ls + '</pre></div></td><td class="code">')
    yield (0, dummyoutfile.getvalue())
    yield (0, '</td></tr></table>')