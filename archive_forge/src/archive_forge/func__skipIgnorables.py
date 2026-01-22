import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def _skipIgnorables(self, instring, loc):
    exprsFound = True
    while exprsFound:
        exprsFound = False
        for e in self.ignoreExprs:
            try:
                while 1:
                    loc, dummy = e._parse(instring, loc)
                    exprsFound = True
            except ParseException:
                pass
    return loc