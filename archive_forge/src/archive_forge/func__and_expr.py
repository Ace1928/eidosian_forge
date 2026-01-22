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
def _and_expr(self):
    expr = self._equality_expr()
    while self.cur_token == 'and':
        self.next_token()
        expr = AndOperator(expr, self._equality_expr())
    return expr