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
class ContainsFunction(Function):
    """The `contains` function, which returns whether a string contains a given
    substring.
    """
    __slots__ = ['string1', 'string2']

    def __init__(self, string1, string2):
        self.string1 = string1
        self.string2 = string2

    def __call__(self, kind, data, pos, namespaces, variables):
        string1 = self.string1(kind, data, pos, namespaces, variables)
        string2 = self.string2(kind, data, pos, namespaces, variables)
        return as_string(string2) in as_string(string1)

    def __repr__(self):
        return 'contains(%r, %r)' % (self.string1, self.string2)