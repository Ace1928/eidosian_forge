import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def writepos(self, pos):
    """Write all params as read in the parse position."""
    result = []
    while not pos.finished():
        if pos.checkskip('$'):
            param = self.writeparam(pos)
            if param:
                result.append(param)
        elif pos.checkskip('f'):
            function = self.writefunction(pos)
            if function:
                function.type = None
                result.append(function)
        elif pos.checkskip('('):
            result.append(self.writebracket('left', '('))
        elif pos.checkskip(')'):
            result.append(self.writebracket('right', ')'))
        else:
            result.append(FormulaConstant(pos.skipcurrent()))
    return result