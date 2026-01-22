from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
class ColumnBase(Node):
    _converter = None

    @Node.copy
    def converter(self, converter=None):
        self._converter = converter

    def alias(self, alias):
        if alias:
            return Alias(self, alias)
        return self

    def unalias(self):
        return self

    def bind_to(self, dest):
        return BindTo(self, dest)

    def cast(self, as_type):
        return Cast(self, as_type)

    def asc(self, collation=None, nulls=None):
        return Asc(self, collation=collation, nulls=nulls)
    __pos__ = asc

    def desc(self, collation=None, nulls=None):
        return Desc(self, collation=collation, nulls=nulls)
    __neg__ = desc

    def __invert__(self):
        return Negated(self)

    def _e(op, inv=False):
        """
        Lightweight factory which returns a method that builds an Expression
        consisting of the left-hand and right-hand operands, using `op`.
        """

        def inner(self, rhs):
            if inv:
                return Expression(rhs, op, self)
            return Expression(self, op, rhs)
        return inner
    __and__ = _e(OP.AND)
    __or__ = _e(OP.OR)
    __add__ = _e(OP.ADD)
    __sub__ = _e(OP.SUB)
    __mul__ = _e(OP.MUL)
    __div__ = __truediv__ = _e(OP.DIV)
    __xor__ = _e(OP.XOR)
    __radd__ = _e(OP.ADD, inv=True)
    __rsub__ = _e(OP.SUB, inv=True)
    __rmul__ = _e(OP.MUL, inv=True)
    __rdiv__ = __rtruediv__ = _e(OP.DIV, inv=True)
    __rand__ = _e(OP.AND, inv=True)
    __ror__ = _e(OP.OR, inv=True)
    __rxor__ = _e(OP.XOR, inv=True)

    def __eq__(self, rhs):
        op = OP.IS if rhs is None else OP.EQ
        return Expression(self, op, rhs)

    def __ne__(self, rhs):
        op = OP.IS_NOT if rhs is None else OP.NE
        return Expression(self, op, rhs)
    __lt__ = _e(OP.LT)
    __le__ = _e(OP.LTE)
    __gt__ = _e(OP.GT)
    __ge__ = _e(OP.GTE)
    __lshift__ = _e(OP.IN)
    __rshift__ = _e(OP.IS)
    __mod__ = _e(OP.LIKE)
    __pow__ = _e(OP.ILIKE)
    like = _e(OP.LIKE)
    ilike = _e(OP.ILIKE)
    bin_and = _e(OP.BIN_AND)
    bin_or = _e(OP.BIN_OR)
    in_ = _e(OP.IN)
    not_in = _e(OP.NOT_IN)
    regexp = _e(OP.REGEXP)
    iregexp = _e(OP.IREGEXP)

    def is_null(self, is_null=True):
        op = OP.IS if is_null else OP.IS_NOT
        return Expression(self, op, None)

    def _escape_like_expr(self, s, template):
        if s.find('_') >= 0 or s.find('%') >= 0 or s.find('\\') >= 0:
            s = s.replace('\\', '\\\\').replace('_', '\\_').replace('%', '\\%')
            return NodeList((Value(template % s, converter=False), SQL('ESCAPE'), Value('\\', converter=False)))
        return template % s

    def contains(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression('%', OP.CONCAT, Expression(rhs, OP.CONCAT, '%'))
        else:
            rhs = self._escape_like_expr(rhs, '%%%s%%')
        return Expression(self, OP.ILIKE, rhs)

    def startswith(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression(rhs, OP.CONCAT, '%')
        else:
            rhs = self._escape_like_expr(rhs, '%s%%')
        return Expression(self, OP.ILIKE, rhs)

    def endswith(self, rhs):
        if isinstance(rhs, Node):
            rhs = Expression('%', OP.CONCAT, rhs)
        else:
            rhs = self._escape_like_expr(rhs, '%%%s')
        return Expression(self, OP.ILIKE, rhs)

    def between(self, lo, hi):
        return Expression(self, OP.BETWEEN, NodeList((lo, SQL('AND'), hi)))

    def concat(self, rhs):
        return StringExpression(self, OP.CONCAT, rhs)

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.start is None or item.stop is None:
                raise ValueError('BETWEEN range must have both a start- and end-point.')
            return self.between(item.start, item.stop)
        return self == item
    __iter__ = None

    def distinct(self):
        return NodeList((SQL('DISTINCT'), self))

    def collate(self, collation):
        return NodeList((self, SQL('COLLATE %s' % collation)))

    def get_sort_key(self, ctx):
        return ()