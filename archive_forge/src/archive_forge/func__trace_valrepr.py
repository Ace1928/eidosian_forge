import copy
import re
from functools import total_ordering
from nltk.internals import raise_unorderable_types, read_str
from nltk.sem.logic import (
def _trace_valrepr(val):
    if isinstance(val, Variable):
        return '%s' % val
    else:
        return '%s' % repr(val)