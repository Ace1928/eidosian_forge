import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def _iterkeys(self):
    if hasattr(self.__tokdict, 'iterkeys'):
        return self.__tokdict.iterkeys()
    else:
        return iter(self.__tokdict)