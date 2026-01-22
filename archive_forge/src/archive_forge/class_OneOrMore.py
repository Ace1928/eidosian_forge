import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class OneOrMore(ParseElementEnhance):
    """Repetition of one or more of the given expression."""

    def parseImpl(self, instring, loc, doActions=True):
        loc, tokens = self.expr._parse(instring, loc, doActions, callPreParse=False)
        try:
            hasIgnoreExprs = len(self.ignoreExprs) > 0
            while 1:
                if hasIgnoreExprs:
                    preloc = self._skipIgnorables(instring, loc)
                else:
                    preloc = loc
                loc, tmptokens = self.expr._parse(instring, preloc, doActions)
                if tmptokens or tmptokens.keys():
                    tokens += tmptokens
        except (ParseException, IndexError):
            pass
        return (loc, tokens)

    def __str__(self):
        if hasattr(self, 'name'):
            return self.name
        if self.strRepr is None:
            self.strRepr = '{' + _ustr(self.expr) + '}...'
        return self.strRepr

    def setResultsName(self, name, listAllMatches=False):
        ret = super(OneOrMore, self).setResultsName(name, listAllMatches)
        ret.saveAsList = True
        return ret