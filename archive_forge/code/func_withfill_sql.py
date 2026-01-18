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
def withfill_sql(self, expression: exp.WithFill) -> str:
    from_sql = self.sql(expression, 'from')
    from_sql = f' FROM {from_sql}' if from_sql else ''
    to_sql = self.sql(expression, 'to')
    to_sql = f' TO {to_sql}' if to_sql else ''
    step_sql = self.sql(expression, 'step')
    step_sql = f' STEP {step_sql}' if step_sql else ''
    return f'WITH FILL{from_sql}{to_sql}{step_sql}'