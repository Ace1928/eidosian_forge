from __future__ import annotations
import datetime
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.time import format_time
from sqlglot.tokens import TokenType
def _parse_projections(self) -> t.List[exp.Expression]:
    """
            T-SQL supports the syntax alias = expression in the SELECT's projection list,
            so we transform all parsed Selects to convert their EQ projections into Aliases.

            See: https://learn.microsoft.com/en-us/sql/t-sql/queries/select-clause-transact-sql?view=sql-server-ver16#syntax
            """
    return [exp.alias_(projection.expression, projection.this.this, copy=False) if isinstance(projection, exp.EQ) and isinstance(projection.this, exp.Column) else projection for projection in super()._parse_projections()]