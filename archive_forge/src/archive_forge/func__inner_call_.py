import keyword
import os
import sys
import token
import tokenize
from IPython.utils.coloransi import TermColors, InputTermColors,ColorScheme, ColorSchemeTable
from .colorable import Colorable
from io import StringIO
def _inner_call_(self, toktype, toktext, start_pos):
    """like call but write to a temporary buffer"""
    buff = StringIO()
    srow, scol = start_pos
    colors = self.colors
    owrite = buff.write
    linesep = os.linesep
    oldpos = self.pos
    newpos = self.lines[srow] + scol
    self.pos = newpos + len(toktext)
    if newpos > oldpos:
        owrite(self.raw[oldpos:newpos])
    if toktype in [token.INDENT, token.DEDENT]:
        self.pos = newpos
        buff.seek(0)
        return buff.read()
    if token.LPAR <= toktype <= token.OP:
        toktype = token.OP
    elif toktype == token.NAME and keyword.iskeyword(toktext):
        toktype = _KEYWORD
    color = colors.get(toktype, colors[_TEXT])
    if linesep in toktext:
        toktext = toktext.replace(linesep, '%s%s%s' % (colors.normal, linesep, color))
    owrite('%s%s%s' % (color, toktext, colors.normal))
    buff.seek(0)
    return buff.read()