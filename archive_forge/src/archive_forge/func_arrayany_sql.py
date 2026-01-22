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
def arrayany_sql(self, expression: exp.ArrayAny) -> str:
    if self.CAN_IMPLEMENT_ARRAY_ANY:
        filtered = exp.ArrayFilter(this=expression.this, expression=expression.expression)
        filtered_not_empty = exp.ArraySize(this=filtered).neq(0)
        original_is_empty = exp.ArraySize(this=expression.this).eq(0)
        return self.sql(exp.paren(original_is_empty.or_(filtered_not_empty)))
    from sqlglot.dialects import Dialect
    if self.dialect.__class__ != Dialect:
        self.unsupported('ARRAY_ANY is unsupported')
    return self.function_fallback_sql(expression)