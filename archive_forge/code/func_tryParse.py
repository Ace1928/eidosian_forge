import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def tryParse(self, instring, loc):
    try:
        return self._parse(instring, loc, doActions=False)[0]
    except ParseFatalException:
        raise ParseException(instring, loc, self.errmsg, self)