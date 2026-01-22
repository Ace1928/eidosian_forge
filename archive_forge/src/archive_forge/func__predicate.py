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
def _predicate(self):
    assert self.cur_token == '['
    self.next_token()
    expr = self._or_expr()
    if self.cur_token != ']':
        raise PathSyntaxError('Expected "]" to close predicate, but found "%s"' % self.cur_token, self.filename, self.lineno)
    if not self.at_end:
        self.next_token()
    return expr