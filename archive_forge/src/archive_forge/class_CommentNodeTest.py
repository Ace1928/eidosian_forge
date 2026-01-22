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
class CommentNodeTest(object):
    """Node test that matches any comment events."""
    __slots__ = []

    def __call__(self, kind, data, pos, namespaces, variables):
        return kind is COMMENT

    def __repr__(self):
        return 'comment()'