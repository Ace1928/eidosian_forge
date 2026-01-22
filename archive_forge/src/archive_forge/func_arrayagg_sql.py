from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.transforms import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def arrayagg_sql(self, expression: exp.ArrayAgg) -> str:
    return self.func('COLLECT_LIST', expression.this.this if isinstance(expression.this, exp.Order) else expression.this)