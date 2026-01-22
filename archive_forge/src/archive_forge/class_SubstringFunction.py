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
class SubstringFunction(Function):
    """The `substring` function that returns the part of a string that starts
    at the given offset, and optionally limited to the given length.
    """
    __slots__ = ['string', 'start', 'length']

    def __init__(self, string, start, length=None):
        self.string = string
        self.start = start
        self.length = length

    def __call__(self, kind, data, pos, namespaces, variables):
        string = self.string(kind, data, pos, namespaces, variables)
        start = self.start(kind, data, pos, namespaces, variables)
        length = 0
        if self.length is not None:
            length = self.length(kind, data, pos, namespaces, variables)
        return string[as_long(start):len(as_string(string)) - as_long(length)]

    def __repr__(self):
        if self.length is not None:
            return 'substring(%r, %r, %r)' % (self.string, self.start, self.length)
        else:
            return 'substring(%r, %r)' % (self.string, self.start)