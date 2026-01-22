from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _uncast_text(self, expression: exp.Expression, name: str) -> str:
    this = expression.this
    if isinstance(this, exp.Cast) and this.is_type(exp.DataType.Type.TEXT):
        this_sql = self.sql(this, 'this')
    else:
        this_sql = self.sql(this)
    expression_sql = self.sql(expression, 'expression')
    return self.func(name, this_sql, expression_sql if expression_sql else None)