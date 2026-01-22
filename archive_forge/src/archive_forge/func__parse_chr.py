from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_chr(self) -> t.Optional[exp.Expression]:
    expressions = self._parse_csv(self._parse_conjunction)
    kwargs: t.Dict[str, t.Any] = {'this': seq_get(expressions, 0)}
    if len(expressions) > 1:
        kwargs['expressions'] = expressions[1:]
    if self._match(TokenType.USING):
        kwargs['charset'] = self._parse_var()
    return self.expression(exp.Chr, **kwargs)