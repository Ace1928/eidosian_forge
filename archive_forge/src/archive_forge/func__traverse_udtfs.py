from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_udtfs(scope):
    if isinstance(scope.expression, exp.Unnest):
        expressions = scope.expression.expressions
    elif isinstance(scope.expression, exp.Lateral):
        expressions = [scope.expression.this]
    else:
        expressions = []
    sources = {}
    for expression in expressions:
        if _is_derived_table(expression):
            top = None
            for child_scope in _traverse_scope(scope.branch(expression, scope_type=ScopeType.DERIVED_TABLE, outer_columns=expression.alias_column_names)):
                yield child_scope
                top = child_scope
                sources[expression.alias] = child_scope
            scope.derived_table_scopes.append(top)
            scope.table_scopes.append(top)
    scope.sources.update(sources)