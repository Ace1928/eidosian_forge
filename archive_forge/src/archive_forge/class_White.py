import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class White(Token):
    """Special matching class for matching whitespace.  Normally, whitespace is ignored
       by pyparsing grammars.  This class is included when some whitespace structures
       are significant.  Define with a string containing the whitespace characters to be
       matched; default is C{" \\t\\r\\n"}.  Also takes optional C{min}, C{max}, and C{exact} arguments,
       as defined for the C{L{Word}} class."""
    whiteStrs = {' ': '<SPC>', '\t': '<TAB>', '\n': '<LF>', '\r': '<CR>', '\x0c': '<FF>'}

    def __init__(self, ws=' \t\r\n', min=1, max=0, exact=0):
        super(White, self).__init__()
        self.matchWhite = ws
        self.setWhitespaceChars(''.join((c for c in self.whiteChars if c not in self.matchWhite)))
        self.name = ''.join((White.whiteStrs[c] for c in self.matchWhite))
        self.mayReturnEmpty = True
        self.errmsg = 'Expected ' + self.name
        self.minLen = min
        if max > 0:
            self.maxLen = max
        else:
            self.maxLen = _MAX_INT
        if exact > 0:
            self.maxLen = exact
            self.minLen = exact

    def parseImpl(self, instring, loc, doActions=True):
        if not instring[loc] in self.matchWhite:
            raise ParseException(instring, loc, self.errmsg, self)
        start = loc
        loc += 1
        maxloc = start + self.maxLen
        maxloc = min(maxloc, len(instring))
        while loc < maxloc and instring[loc] in self.matchWhite:
            loc += 1
        if loc - start < self.minLen:
            raise ParseException(instring, loc, self.errmsg, self)
        return (loc, instring[start:loc])