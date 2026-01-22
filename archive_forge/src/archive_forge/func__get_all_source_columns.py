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
def _get_all_source_columns(self) -> t.Dict[str, t.Sequence[str]]:
    if self._source_columns is None:
        self._source_columns = {source_name: self.get_source_columns(source_name) for source_name, source in itertools.chain(self.scope.selected_sources.items(), self.scope.lateral_sources.items())}
    return self._source_columns