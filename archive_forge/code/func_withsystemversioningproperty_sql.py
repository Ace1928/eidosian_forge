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
def withsystemversioningproperty_sql(self, expression: exp.WithSystemVersioningProperty) -> str:
    sql = 'WITH(SYSTEM_VERSIONING=ON'
    if expression.this:
        history_table = self.sql(expression, 'this')
        sql = f'{sql}(HISTORY_TABLE={history_table}'
        if expression.expression:
            data_consistency_check = self.sql(expression, 'expression')
            sql = f'{sql}, DATA_CONSISTENCY_CHECK={data_consistency_check}'
        sql = f'{sql})'
    return f'{sql})'