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
class StringLiteral(Literal):
    """A string literal node."""
    __slots__ = ['text']

    def __init__(self, text):
        self.text = text

    def __call__(self, kind, data, pos, namespaces, variables):
        return self.text

    def __repr__(self):
        return '"%s"' % self.text