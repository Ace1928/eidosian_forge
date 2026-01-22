import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class FollowedBy(ParseElementEnhance):
    """Lookahead matching of the given parse expression.  C{FollowedBy}
    does *not* advance the parsing position within the input string, it only
    verifies that the specified parse expression matches at the current
    position.  C{FollowedBy} always returns a null token list."""

    def __init__(self, expr):
        super(FollowedBy, self).__init__(expr)
        self.mayReturnEmpty = True

    def parseImpl(self, instring, loc, doActions=True):
        self.expr.tryParse(instring, loc)
        return (loc, [])