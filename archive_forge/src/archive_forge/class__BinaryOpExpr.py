from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _BinaryOpExpr(_FuncExpr):

    def __init__(self, op: str, left: Any, right: Any, arg_distinct: bool=False):
        super().__init__(op, _to_col(left), _to_col(right), arg_distinct=arg_distinct)

    @property
    def left(self) -> ColumnExpr:
        return self.args[0]

    @property
    def right(self) -> ColumnExpr:
        return self.args[1]

    @property
    def op(self) -> str:
        return self.func

    def _copy(self) -> _FuncExpr:
        return _BinaryOpExpr(self.op, self.left, self.right)