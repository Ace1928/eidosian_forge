from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def tonumber_sql(self, expression: exp.ToNumber) -> str:
    return self.func('TO_NUMBER', expression.this, expression.args.get('format'), expression.args.get('precision'), expression.args.get('scale'))