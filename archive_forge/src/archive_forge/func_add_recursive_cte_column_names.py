from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def add_recursive_cte_column_names(expression: exp.Expression) -> exp.Expression:
    """Uses projection output names in recursive CTE definitions to define the CTEs' columns."""
    if isinstance(expression, exp.With) and expression.recursive:
        next_name = name_sequence('_c_')
        for cte in expression.expressions:
            if not cte.args['alias'].columns:
                query = cte.this
                if isinstance(query, exp.Union):
                    query = query.this
                cte.args['alias'].set('columns', [exp.to_identifier(s.alias_or_name or next_name()) for s in query.selects])
    return expression