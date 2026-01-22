from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _pushdown_cte_column_names(expression: exp.Expression) -> exp.Expression:
    """BigQuery doesn't allow column names when defining a CTE, so we try to push them down."""
    if isinstance(expression, exp.CTE) and expression.alias_column_names:
        cte_query = expression.this
        if cte_query.is_star:
            logger.warning("Can't push down CTE column names for star queries. Run the query through the optimizer or use 'qualify' to expand the star projections first.")
            return expression
        column_names = expression.alias_column_names
        expression.args['alias'].set('columns', None)
        for name, select in zip(column_names, cte_query.selects):
            to_replace = select
            if isinstance(select, exp.Alias):
                select = select.this
            to_replace.replace(exp.alias_(select, name))
    return expression