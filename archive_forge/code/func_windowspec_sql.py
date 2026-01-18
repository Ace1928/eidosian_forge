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
def windowspec_sql(self, expression: exp.WindowSpec) -> str:
    kind = self.sql(expression, 'kind')
    start = csv(self.sql(expression, 'start'), self.sql(expression, 'start_side'), sep=' ')
    end = csv(self.sql(expression, 'end'), self.sql(expression, 'end_side'), sep=' ') or 'CURRENT ROW'
    return f'{kind} BETWEEN {start} AND {end}'