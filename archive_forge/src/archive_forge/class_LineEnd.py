import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class LineEnd(_PositionToken):
    """Matches if current position is at the end of a line within the parse string"""

    def __init__(self):
        super(LineEnd, self).__init__()
        self.setWhitespaceChars(ParserElement.DEFAULT_WHITE_CHARS.replace('\n', ''))
        self.errmsg = 'Expected end of line'

    def parseImpl(self, instring, loc, doActions=True):
        if loc < len(instring):
            if instring[loc] == '\n':
                return (loc + 1, '\n')
            else:
                raise ParseException(instring, loc, self.errmsg, self)
        elif loc == len(instring):
            return (loc + 1, [])
        else:
            raise ParseException(instring, loc, self.errmsg, self)