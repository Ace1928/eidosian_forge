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
def _select_by_pos(scope: Scope, node: exp.Literal) -> exp.Alias:
    try:
        return scope.expression.selects[int(node.this) - 1].assert_is(exp.Alias)
    except IndexError:
        raise OptimizeError(f'Unknown output column: {node.name}')