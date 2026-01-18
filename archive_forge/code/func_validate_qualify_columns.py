from __future__ import annotations
import itertools
import typing as t
from sqlglot import alias, exp
from sqlglot.dialects.dialect import Dialect, DialectType
from sqlglot.errors import OptimizeError
from sqlglot.helper import seq_get, SingleValuedMapping
from sqlglot.optimizer.annotate_types import TypeAnnotator
from sqlglot.optimizer.scope import Scope, build_scope, traverse_scope, walk_in_scope
from sqlglot.optimizer.simplify import simplify_parens
from sqlglot.schema import Schema, ensure_schema
def validate_qualify_columns(expression: E) -> E:
    """Raise an `OptimizeError` if any columns aren't qualified"""
    all_unqualified_columns = []
    for scope in traverse_scope(expression):
        if isinstance(scope.expression, exp.Select):
            unqualified_columns = scope.unqualified_columns
            if scope.external_columns and (not scope.is_correlated_subquery) and (not scope.pivots):
                column = scope.external_columns[0]
                for_table = f" for table: '{column.table}'" if column.table else ''
                raise OptimizeError(f"Column '{column}' could not be resolved{for_table}")
            if unqualified_columns and scope.pivots and scope.pivots[0].unpivot:
                unpivot_columns = set(_unpivot_columns(scope.pivots[0]))
                unqualified_columns = [c for c in unqualified_columns if c not in unpivot_columns]
            all_unqualified_columns.extend(unqualified_columns)
    if all_unqualified_columns:
        raise OptimizeError(f'Ambiguous columns: {all_unqualified_columns}')
    return expression