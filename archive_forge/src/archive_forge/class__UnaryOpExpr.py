from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _UnaryOpExpr(_FuncExpr):

    def __init__(self, op: str, column: ColumnExpr, arg_distinct: bool=False):
        super().__init__(op, column, arg_distinct=arg_distinct)

    @property
    def col(self) -> ColumnExpr:
        return self.args[0]

    @property
    def op(self) -> str:
        return self.func

    def infer_alias(self) -> ColumnExpr:
        return self if self.output_name != '' else self.alias(self.col.infer_alias().output_name)

    def _copy(self) -> _FuncExpr:
        return _UnaryOpExpr(self.op, self.col)