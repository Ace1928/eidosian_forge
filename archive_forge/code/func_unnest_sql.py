from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def unnest_sql(self, expression: exp.Unnest) -> str:
    if len(expression.expressions) == 1:
        from sqlglot.optimizer.annotate_types import annotate_types
        this = annotate_types(expression.expressions[0])
        if this.is_type('array<json>'):
            while isinstance(this, exp.Cast):
                this = this.this
            arg = self.sql(exp.cast(this, exp.DataType.Type.JSON))
            alias = self.sql(expression, 'alias')
            alias = f' AS {alias}' if alias else ''
            if expression.args.get('offset'):
                self.unsupported('Unsupported JSON_ARRAY_ELEMENTS with offset')
            return f'JSON_ARRAY_ELEMENTS({arg}){alias}'
    return super().unnest_sql(expression)