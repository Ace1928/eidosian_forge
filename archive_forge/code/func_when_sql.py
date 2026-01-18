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
def when_sql(self, expression: exp.When) -> str:
    matched = 'MATCHED' if expression.args['matched'] else 'NOT MATCHED'
    source = ' BY SOURCE' if self.MATCHED_BY_SOURCE and expression.args.get('source') else ''
    condition = self.sql(expression, 'condition')
    condition = f' AND {condition}' if condition else ''
    then_expression = expression.args.get('then')
    if isinstance(then_expression, exp.Insert):
        then = f'INSERT {self.sql(then_expression, 'this')}'
        if 'expression' in then_expression.args:
            then += f' VALUES {self.sql(then_expression, 'expression')}'
    elif isinstance(then_expression, exp.Update):
        if isinstance(then_expression.args.get('expressions'), exp.Star):
            then = f'UPDATE {self.sql(then_expression, 'expressions')}'
        else:
            then = f'UPDATE SET {self.expressions(then_expression, flat=True)}'
    else:
        then = self.sql(then_expression)
    return f'WHEN {matched}{source}{condition} THEN {then}'