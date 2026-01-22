from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _InvertOpExpr(_UnaryOpExpr):

    def _copy(self) -> _FuncExpr:
        return _InvertOpExpr(self.op, self.col)

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        if self.as_type is not None:
            return self.as_type
        tp = self.col.infer_type(schema)
        if pa.types.is_signed_integer(tp) or pa.types.is_floating(tp):
            return tp
        return None