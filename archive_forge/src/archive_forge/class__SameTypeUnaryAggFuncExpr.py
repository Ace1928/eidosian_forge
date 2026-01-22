from typing import Any, Optional
import pyarrow as pa
from fugue.column.expressions import (
from triad import Schema
class _SameTypeUnaryAggFuncExpr(_UnaryAggFuncExpr):

    def _copy(self) -> _FuncExpr:
        return _SameTypeUnaryAggFuncExpr(self.func, *self.args, **self.kwargs)

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        return self.as_type or self.args[0].infer_type(schema)