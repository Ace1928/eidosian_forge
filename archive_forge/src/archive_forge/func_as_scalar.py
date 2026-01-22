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
def as_scalar(value):
    """Convert value to a scalar. If a single element Attrs() object is passed
    the value of the single attribute will be returned."""
    if isinstance(value, Attrs):
        assert len(value) == 1
        return value[0][1]
    else:
        return value