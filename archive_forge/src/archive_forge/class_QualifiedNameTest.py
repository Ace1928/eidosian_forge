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
class QualifiedNameTest(object):
    """Node test that matches any event with the given principal type and
    qualified name.
    """
    __slots__ = ['principal_type', 'prefix', 'name']

    def __init__(self, principal_type, prefix, name):
        self.principal_type = principal_type
        self.prefix = prefix
        self.name = name

    def __call__(self, kind, data, pos, namespaces, variables):
        qname = QName('%s}%s' % (namespaces.get(self.prefix), self.name))
        if kind is START:
            if self.principal_type is ATTRIBUTE and qname in data[1]:
                return Attrs([(qname, data[1].get(qname))])
            else:
                return data[0] == qname

    def __repr__(self):
        return '%s:%s' % (self.prefix, self.name)