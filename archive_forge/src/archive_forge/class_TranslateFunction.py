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
class TranslateFunction(Function):
    """The `translate` function that translates a set of characters in a
    string to target set of characters.
    """
    __slots__ = ['string', 'fromchars', 'tochars']

    def __init__(self, string, fromchars, tochars):
        self.string = string
        self.fromchars = fromchars
        self.tochars = tochars

    def __call__(self, kind, data, pos, namespaces, variables):
        string = as_string(self.string(kind, data, pos, namespaces, variables))
        fromchars = as_string(self.fromchars(kind, data, pos, namespaces, variables))
        tochars = as_string(self.tochars(kind, data, pos, namespaces, variables))
        table = dict(zip([ord(c) for c in fromchars], [ord(c) for c in tochars]))
        return string.translate(table)

    def __repr__(self):
        return 'translate(%r, %r, %r)' % (self.string, self.fromchars, self.tochars)