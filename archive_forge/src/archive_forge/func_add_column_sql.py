from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def add_column_sql(self, expression: exp.AlterTable) -> str:
    actions = self.expressions(expression, key='actions', flat=True)
    if len(expression.args.get('actions', [])) > 1:
        return f'ADD ({actions})'
    return f'ADD {actions}'