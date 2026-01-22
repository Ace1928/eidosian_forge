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
def _embed_ignore_nulls(self, expression: exp.IgnoreNulls | exp.RespectNulls, text: str) -> str:
    if self.IGNORE_NULLS_IN_FUNC and (not expression.meta.get('inline')):
        mods = sorted(expression.find_all(exp.HavingMax, exp.Order, exp.Limit), key=lambda x: 0 if isinstance(x, exp.HavingMax) else 1 if isinstance(x, exp.Order) else 2)
        if mods:
            mod = mods[0]
            this = expression.__class__(this=mod.this.copy())
            this.meta['inline'] = True
            mod.this.replace(this)
            return self.sql(expression.this)
        agg_func = expression.find(exp.AggFunc)
        if agg_func:
            return self.sql(agg_func)[:-1] + f' {text})'
    return f'{self.sql(expression, 'this')} {text}'