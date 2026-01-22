from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.postgres import Postgres
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_table(self, schema: bool=False, joins: bool=False, alias_tokens: t.Optional[t.Collection[TokenType]]=None, parse_bracket: bool=False, is_db_reference: bool=False, parse_partition: bool=False) -> t.Optional[exp.Expression]:
    unpivot = self._match(TokenType.UNPIVOT)
    table = super()._parse_table(schema=schema, joins=joins, alias_tokens=alias_tokens, parse_bracket=parse_bracket, is_db_reference=is_db_reference)
    return self.expression(exp.Pivot, this=table, unpivot=True) if unpivot else table