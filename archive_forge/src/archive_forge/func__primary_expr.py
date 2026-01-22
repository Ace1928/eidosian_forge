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
def _primary_expr(self):
    token = self.cur_token
    if len(token) > 1 and (token[0], token[-1]) in self._QUOTES:
        self.next_token()
        return StringLiteral(token[1:-1])
    elif token[0].isdigit() or token[0] == '.':
        self.next_token()
        return NumberLiteral(as_float(token))
    elif token == '$':
        token = self.next_token()
        self.next_token()
        return VariableReference(token)
    elif not self.at_end and self.peek_token().startswith('('):
        return self._function_call()
    else:
        axis = None
        if token == '@':
            axis = ATTRIBUTE
            self.next_token()
        return self._node_test(axis)