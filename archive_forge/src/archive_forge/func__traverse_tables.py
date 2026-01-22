from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def _traverse_tables(scope):
    sources = {}
    expressions = []
    from_ = scope.expression.args.get('from')
    if from_:
        expressions.append(from_.this)
    for join in scope.expression.args.get('joins') or []:
        expressions.append(join.this)
    if isinstance(scope.expression, exp.Table):
        expressions.append(scope.expression)
    expressions.extend(scope.expression.args.get('laterals') or [])
    for expression in expressions:
        if isinstance(expression, exp.Table):
            table_name = expression.name
            source_name = expression.alias_or_name
            if table_name in scope.sources and (not expression.db):
                pivots = expression.args.get('pivots')
                if pivots:
                    sources[pivots[0].alias] = expression
                else:
                    sources[source_name] = scope.sources[table_name]
            elif source_name in sources:
                sources[find_new_name(sources, table_name)] = expression
            else:
                sources[source_name] = expression
            if expression is not scope.expression:
                expressions.extend((join.this for join in expression.args.get('joins') or []))
            continue
        if not isinstance(expression, exp.DerivedTable):
            continue
        if isinstance(expression, exp.UDTF):
            lateral_sources = sources
            scope_type = ScopeType.UDTF
            scopes = scope.udtf_scopes
        elif _is_derived_table(expression):
            lateral_sources = None
            scope_type = ScopeType.DERIVED_TABLE
            scopes = scope.derived_table_scopes
            expressions.extend((join.this for join in expression.args.get('joins') or []))
        else:
            expressions.append(expression.this)
            expressions.extend((join.this for join in expression.args.get('joins') or []))
            continue
        for child_scope in _traverse_scope(scope.branch(expression, lateral_sources=lateral_sources, outer_columns=expression.alias_column_names, scope_type=scope_type)):
            yield child_scope
            sources[expression.alias] = child_scope
        scopes.append(child_scope)
        scope.table_scopes.append(child_scope)
    scope.sources.update(sources)