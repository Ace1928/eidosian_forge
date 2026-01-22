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
class CompoundSelectQuery(SelectBase):

    def __init__(self, lhs, op, rhs):
        super(CompoundSelectQuery, self).__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs

    @property
    def _returning(self):
        return self.lhs._returning

    @database_required
    def exists(self, database):
        query = Select((self.limit(1),), (SQL('1'),)).bind(database)
        return bool(query.scalar())

    def _get_query_key(self):
        return (self.lhs.get_query_key(), self.rhs.get_query_key())

    def _wrap_parens(self, ctx, subq):
        csq_setting = ctx.state.compound_select_parentheses
        if not csq_setting or csq_setting == CSQ_PARENTHESES_NEVER:
            return False
        elif csq_setting == CSQ_PARENTHESES_ALWAYS:
            return True
        elif csq_setting == CSQ_PARENTHESES_UNNESTED:
            if ctx.state.in_expr or ctx.state.in_function:
                return False
            return not isinstance(subq, CompoundSelectQuery)

    def __sql__(self, ctx):
        if ctx.scope == SCOPE_COLUMN:
            return self.apply_column(ctx)
        super(CompoundSelectQuery, self).__sql__(ctx)
        outer_parens = ctx.subquery or ctx.scope == SCOPE_SOURCE
        with ctx(parentheses=outer_parens):
            lhs_parens = self._wrap_parens(ctx, self.lhs)
            with ctx.scope_normal(parentheses=lhs_parens, subquery=False):
                ctx.sql(self.lhs)
            ctx.literal(' %s ' % self.op)
            with ctx.push_alias():
                rhs_parens = self._wrap_parens(ctx, self.rhs)
                with ctx.scope_normal(parentheses=rhs_parens, subquery=False):
                    ctx.sql(self.rhs)
            with ctx.scope_values():
                self._apply_ordering(ctx)
        return self.apply_alias(ctx)