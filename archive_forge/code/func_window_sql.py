from __future__ import annotations
import logging
import re
import typing as t
from collections import defaultdict
from functools import reduce
from sqlglot import exp
from sqlglot.errors import ErrorLevel, UnsupportedError, concat_messages
from sqlglot.helper import apply_index_offset, csv, seq_get
from sqlglot.jsonpath import ALL_JSON_PATH_PARTS, JSON_PATH_PART_TRANSFORMS
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def window_sql(self, expression: exp.Window) -> str:
    this = self.sql(expression, 'this')
    partition = self.partition_by_sql(expression)
    order = expression.args.get('order')
    order = self.order_sql(order, flat=True) if order else ''
    spec = self.sql(expression, 'spec')
    alias = self.sql(expression, 'alias')
    over = self.sql(expression, 'over') or 'OVER'
    this = f'{this} {('AS' if expression.arg_key == 'windows' else over)}'
    first = expression.args.get('first')
    if first is None:
        first = ''
    else:
        first = 'FIRST' if first else 'LAST'
    if not partition and (not order) and (not spec) and alias:
        return f'{this} {alias}'
    args = ' '.join((arg for arg in (alias, first, partition, order, spec) if arg))
    return f'{this} ({args})'