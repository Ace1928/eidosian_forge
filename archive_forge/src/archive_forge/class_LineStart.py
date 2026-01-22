import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class LineStart(_PositionToken):
    """Matches if current position is at the beginning of a line within the parse string"""

    def __init__(self):
        super(LineStart, self).__init__()
        self.setWhitespaceChars(ParserElement.DEFAULT_WHITE_CHARS.replace('\n', ''))
        self.errmsg = 'Expected start of line'

    def preParse(self, instring, loc):
        preloc = super(LineStart, self).preParse(instring, loc)
        if instring[preloc] == '\n':
            loc += 1
        return loc

    def parseImpl(self, instring, loc, doActions=True):
        if not (loc == 0 or loc == self.preParse(instring, 0) or instring[loc - 1] == '\n'):
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])