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
def _unpivot_columns(unpivot: exp.Pivot) -> t.Iterator[exp.Column]:
    name_column = []
    field = unpivot.args.get('field')
    if isinstance(field, exp.In) and isinstance(field.this, exp.Column):
        name_column.append(field.this)
    value_columns = (c for e in unpivot.expressions for c in e.find_all(exp.Column))
    return itertools.chain(name_column, value_columns)