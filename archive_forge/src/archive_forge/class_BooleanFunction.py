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
class BooleanFunction(Function):
    """The `boolean` function, which converts its argument to a boolean
    value.
    """
    __slots__ = ['expr']
    _return_type = bool

    def __init__(self, expr):
        self.expr = expr

    def __call__(self, kind, data, pos, namespaces, variables):
        val = self.expr(kind, data, pos, namespaces, variables)
        return as_bool(val)

    def __repr__(self):
        return 'boolean(%r)' % self.expr