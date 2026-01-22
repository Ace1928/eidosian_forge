from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_with_type_lambda(data_type: exp.DataType.Type) -> t.Callable[[TypeAnnotator, E], E]:
    return lambda self, e: self._annotate_with_type(e, data_type)