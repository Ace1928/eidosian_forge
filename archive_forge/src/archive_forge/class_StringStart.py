import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class StringStart(_PositionToken):
    """Matches if current position is at the beginning of the parse string"""

    def __init__(self):
        super(StringStart, self).__init__()
        self.errmsg = 'Expected start of text'

    def parseImpl(self, instring, loc, doActions=True):
        if loc != 0:
            if loc != self.preParse(instring, 0):
                raise ParseException(instring, loc, self.errmsg, self)
        return (loc, [])