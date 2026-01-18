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
def truncatetable_sql(self, expression: exp.TruncateTable) -> str:
    target = 'DATABASE' if expression.args.get('is_database') else 'TABLE'
    tables = f' {self.expressions(expression)}'
    exists = ' IF EXISTS' if expression.args.get('exists') else ''
    on_cluster = self.sql(expression, 'cluster')
    on_cluster = f' {on_cluster}' if on_cluster else ''
    identity = self.sql(expression, 'identity')
    identity = f' {identity} IDENTITY' if identity else ''
    option = self.sql(expression, 'option')
    option = f' {option}' if option else ''
    partition = self.sql(expression, 'partition')
    partition = f' {partition}' if partition else ''
    return f'TRUNCATE {target}{exists}{tables}{on_cluster}{identity}{option}{partition}'