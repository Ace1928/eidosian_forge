import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def __lookup(self, sub):
    for k, vlist in self.__tokdict.items():
        for v, loc in vlist:
            if sub is v:
                return k
    return None