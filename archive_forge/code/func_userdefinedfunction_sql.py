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
def userdefinedfunction_sql(self, expression: exp.UserDefinedFunction) -> str:
    this = self.sql(expression, 'this')
    expressions = self.no_identify(self.expressions, expression)
    expressions = self.wrap(expressions) if expression.args.get('wrapped') else f' {expressions}'
    return f'{this}{expressions}'