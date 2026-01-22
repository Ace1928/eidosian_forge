import _collections_abc
import sys as _sys
from itertools import chain as _chain
from itertools import repeat as _repeat
from itertools import starmap as _starmap
from keyword import iskeyword as _iskeyword
from operator import eq as _eq
from operator import itemgetter as _itemgetter
from reprlib import recursive_repr as _recursive_repr
from _weakref import proxy as _proxy
def _keep_positive(self):
    """Internal method to strip elements with a negative or zero count"""
    nonpositive = [elem for elem, count in self.items() if not count > 0]
    for elem in nonpositive:
        del self[elem]
    return self