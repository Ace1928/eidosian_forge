from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def after_limit_modifiers(self, expression: exp.Expression) -> t.List[str]:
    return super().after_limit_modifiers(expression) + [self.seg('SETTINGS ') + self.expressions(expression, key='settings', flat=True) if expression.args.get('settings') else '', self.seg('FORMAT ') + self.sql(expression, 'format') if expression.args.get('format') else '']