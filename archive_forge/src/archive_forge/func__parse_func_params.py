from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_func_params(self, this: t.Optional[exp.Func]=None) -> t.Optional[t.List[exp.Expression]]:
    if self._match_pair(TokenType.R_PAREN, TokenType.L_PAREN):
        return self._parse_csv(self._parse_lambda)
    if self._match(TokenType.L_PAREN):
        params = self._parse_csv(self._parse_lambda)
        self._match_r_paren(this)
        return params
    return None