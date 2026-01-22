from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _NotOpExpr(_UnaryOpExpr):

    def _copy(self) -> _FuncExpr:
        return _NotOpExpr(self.op, self.col)

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        if self.as_type is not None:
            return self.as_type
        tp = self.col.infer_type(schema)
        if pa.types.is_boolean(tp):
            return tp
        return None