from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression
from triad.utils.schema import quote_name
from fugue.column.expressions import (
from fugue.column.functions import is_agg
from fugue.exceptions import FugueBug
def _on_common_func(self, expr: _FuncExpr) -> Iterable[str]:

    def to_str(v: Any) -> Iterable[str]:
        if isinstance(v, ColumnExpr):
            yield from self._generate(v)
        else:
            yield from self._generate(lit(v))

    def get_args() -> Iterable[str]:
        for x in expr.args:
            yield from to_str(x)
            yield ','
        for k, v in expr.kwargs.items():
            yield k
            yield '='
            yield from to_str(v)
            yield ','
    args = list(get_args())
    if len(args) > 0:
        args = args[:-1]
    yield expr.func
    yield '('
    if expr.is_distinct:
        yield 'DISTINCT '
    yield from args
    yield ')'