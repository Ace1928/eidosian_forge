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
class LocalNameTest(object):
    """Node test that matches any event with the given principal type and
    local name.
    """
    __slots__ = ['principal_type', 'name']

    def __init__(self, principal_type, name):
        self.principal_type = principal_type
        self.name = name

    def __call__(self, kind, data, pos, namespaces, variables):
        if kind is START:
            if self.principal_type is ATTRIBUTE and self.name in data[1]:
                return Attrs([(self.name, data[1].get(self.name))])
            else:
                return data[0].localname == self.name

    def __repr__(self):
        return self.name