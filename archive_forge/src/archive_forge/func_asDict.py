import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
def asDict(self):
    """Returns the named parse results as dictionary."""
    return dict(self.items())