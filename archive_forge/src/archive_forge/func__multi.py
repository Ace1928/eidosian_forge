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
def _multi(event, namespaces, variables, updateonly=False):
    retval = None
    for test in tests:
        val = test(event, namespaces, variables, updateonly=updateonly)
        if retval is None:
            retval = val
    return retval