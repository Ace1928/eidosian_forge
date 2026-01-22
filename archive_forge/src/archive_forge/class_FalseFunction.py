from collections import deque
from functools import reduce
from math import ceil, floor
import operator
import re
from itertools import chain
import six
from genshi.compat import IS_PYTHON2
from genshi.core import Stream, Attrs, Namespace, QName
from genshi.core import START, END, TEXT, START_NS, END_NS, COMMENT, PI, \
class FalseFunction(Function):
    """The `false` function, which always returns the boolean `false` value."""
    __slots__ = []

    def __call__(self, kind, data, pos, namespaces, variables):
        return False

    def __repr__(self):
        return 'false()'