import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def transformString(self, instring):
    """Extension to C{L{scanString}}, to modify matching text with modified tokens that may
           be returned from a parse action.  To use C{transformString}, define a grammar and
           attach a parse action to it that modifies the returned token list.
           Invoking C{transformString()} on a target string will then scan for matches,
           and replace the matched text patterns according to the logic in the parse
           action.  C{transformString()} returns the resulting transformed string."""
    out = []
    lastE = 0
    self.keepTabs = True
    try:
        for t, s, e in self.scanString(instring):
            out.append(instring[lastE:s])
            if t:
                if isinstance(t, ParseResults):
                    out += t.asList()
                elif isinstance(t, list):
                    out += t
                else:
                    out.append(t)
            lastE = e
        out.append(instring[lastE:])
        out = [o for o in out if o]
        return ''.join(map(_ustr, _flatten(out)))
    except ParseBaseException as exc:
        if ParserElement.verbose_stacktrace:
            raise
        else:
            raise exc