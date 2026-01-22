from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _show_parser(*args: t.Any, **kwargs: t.Any) -> t.Callable[[MySQL.Parser], exp.Show]:

    def _parse(self: MySQL.Parser) -> exp.Show:
        return self._parse_show_mysql(*args, **kwargs)
    return _parse