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
class SubstringAfterFunction(Function):
    """The `substring-after` function that returns the part of a string that
    is found after the given substring.
    """
    __slots__ = ['string1', 'string2']

    def __init__(self, string1, string2):
        self.string1 = string1
        self.string2 = string2

    def __call__(self, kind, data, pos, namespaces, variables):
        string1 = as_string(self.string1(kind, data, pos, namespaces, variables))
        string2 = as_string(self.string2(kind, data, pos, namespaces, variables))
        index = string1.find(string2)
        if index >= 0:
            return string1[index + len(string2):]
        return ''

    def __repr__(self):
        return 'substring-after(%r, %r)' % (self.string1, self.string2)