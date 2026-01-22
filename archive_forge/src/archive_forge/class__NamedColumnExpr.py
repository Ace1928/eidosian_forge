from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _NamedColumnExpr(ColumnExpr):

    def __init__(self, name: Any):
        self._name = name
        super().__init__()

    @property
    def body_str(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_name(self) -> str:
        return super().output_name

    def alias(self, as_name: str) -> ColumnExpr:
        other = _NamedColumnExpr(self.name)
        other._as_name = as_name
        other._as_type = self.as_type
        return other

    def cast(self, data_type: Any) -> 'ColumnExpr':
        other = _NamedColumnExpr(self.name)
        other._as_name = self.as_name
        other._as_type = None if data_type is None else to_pa_datatype(data_type)
        return other

    def infer_alias(self) -> ColumnExpr:
        if self.as_name == '' and self.as_type is not None:
            return self.alias(self.output_name)
        return self

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        if self.name not in schema:
            return self.as_type
        return self.as_type or schema[self.name].type

    def _uuid_keys(self) -> List[Any]:
        return [self.name]