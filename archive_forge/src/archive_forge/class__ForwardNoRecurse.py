import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class _ForwardNoRecurse(Forward):

    def __str__(self):
        return '...'