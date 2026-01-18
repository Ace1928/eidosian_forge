from __future__ import annotations
import typing as t
from sqlglot import exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.optimizer.isolate_table_selects import isolate_table_selects
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.qualify_columns import (
from sqlglot.optimizer.qualify_tables import qualify_tables
from sqlglot.schema import Schema, ensure_schema

    Rewrite sqlglot AST to have normalized and qualified tables and columns.

    This step is necessary for all further SQLGlot optimizations.

    Example:
        >>> import sqlglot
        >>> schema = {"tbl": {"col": "INT"}}
        >>> expression = sqlglot.parse_one("SELECT col FROM tbl")
        >>> qualify(expression, schema=schema).sql()
        'SELECT "tbl"."col" AS "col" FROM "tbl" AS "tbl"'

    Args:
        expression: Expression to qualify.
        db: Default database name for tables.
        catalog: Default catalog name for tables.
        schema: Schema to infer column names and types.
        expand_alias_refs: Whether to expand references to aliases.
        expand_stars: Whether to expand star queries. This is a necessary step
            for most of the optimizer's rules to work; do not set to False unless you
            know what you're doing!
        infer_schema: Whether to infer the schema if missing.
        isolate_tables: Whether to isolate table selects.
        qualify_columns: Whether to qualify columns.
        validate_qualify_columns: Whether to validate columns.
        quote_identifiers: Whether to run the quote_identifiers step.
            This step is necessary to ensure correctness for case sensitive queries.
            But this flag is provided in case this step is performed at a later time.
        identify: If True, quote all identifiers, else only necessary ones.

    Returns:
        The qualified expression.
    