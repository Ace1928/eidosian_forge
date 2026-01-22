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
@property
def all_columns(self) -> t.Set[str]:
    """All available columns of all sources in this scope"""
    if self._all_columns is None:
        self._all_columns = {column for columns in self._get_all_source_columns().values() for column in columns}
    return self._all_columns