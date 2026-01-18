from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType

    Snowflake doesn't allow columns referenced in UNPIVOT to be qualified,
    so we need to unqualify them.

    Example:
        >>> from sqlglot import parse_one
        >>> expr = parse_one("SELECT * FROM m_sales UNPIVOT(sales FOR month IN (m_sales.jan, feb, mar, april))")
        >>> print(_unqualify_unpivot_columns(expr).sql(dialect="snowflake"))
        SELECT * FROM m_sales UNPIVOT(sales FOR month IN (jan, feb, mar, april))
    