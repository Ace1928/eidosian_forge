from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def _on_common_binary(self, expr: _BinaryOpExpr, bracket: bool) -> Iterable[str]:
    assert_or_throw(expr.op in _SUPPORTED_OPERATORS, NotImplementedError(expr))
    if bracket:
        yield '('
    if expr.is_distinct:
        raise FugueBug(f'impossible case {expr}')
    yield from self._generate(expr.left, bracket=True)
    yield _SUPPORTED_OPERATORS[expr.op]
    yield from self._generate(expr.right, bracket=True)
    if bracket:
        yield ')'