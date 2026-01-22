from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.postgres import Postgres
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_approximate_count(self) -> t.Optional[exp.ApproxDistinct]:
    index = self._index - 1
    func = self._parse_function()
    if isinstance(func, exp.Count) and isinstance(func.this, exp.Distinct):
        return self.expression(exp.ApproxDistinct, this=seq_get(func.this.expressions, 0))
    self._retreat(index)
    return None