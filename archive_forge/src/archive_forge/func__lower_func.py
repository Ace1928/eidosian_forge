from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _lower_func(sql: str) -> str:
    index = sql.index('(')
    return sql[:index].lower() + sql[index:]