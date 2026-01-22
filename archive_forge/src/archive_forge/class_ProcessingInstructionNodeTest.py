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
class ProcessingInstructionNodeTest(object):
    """Node test that matches any processing instruction event."""
    __slots__ = ['target']

    def __init__(self, target=None):
        self.target = target

    def __call__(self, kind, data, pos, namespaces, variables):
        return kind is PI and (not self.target or data[0] == self.target)

    def __repr__(self):
        arg = ''
        if self.target:
            arg = '"' + self.target + '"'
        return 'processing-instruction(%s)' % arg