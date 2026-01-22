from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class _LiteralColumnExpr(ColumnExpr):
    _VALID_TYPES = (int, bool, float, str)

    def __init__(self, value: Any):
        assert_or_throw(value is None or isinstance(value, _LiteralColumnExpr._VALID_TYPES), lambda: NotImplementedError(f'{value}, type: {type(value)}'))
        self._value = value
        super().__init__()

    @property
    def body_str(self) -> str:
        if self.value is None:
            return 'NULL'
        elif isinstance(self.value, str):
            body = self.value.translate(str.maketrans({'\\': '\\\\', "'": "\\'"}))
            return f"'{body}'"
        elif isinstance(self.value, bool):
            return 'TRUE' if self.value else 'FALSE'
        else:
            return str(self.value)

    @property
    def value(self) -> Any:
        return self._value

    def is_null(self) -> ColumnExpr:
        return _LiteralColumnExpr(self.value is None)

    def not_null(self) -> ColumnExpr:
        return _LiteralColumnExpr(self.value is not None)

    def alias(self, as_name: str) -> ColumnExpr:
        other = _LiteralColumnExpr(self.value)
        other._as_name = as_name
        other._as_type = self.as_type
        return other

    def cast(self, data_type: Any) -> ColumnExpr:
        other = _LiteralColumnExpr(self.value)
        other._as_name = self.as_name
        other._as_type = None if data_type is None else to_pa_datatype(data_type)
        return other

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        if self.value is None:
            return self.as_type
        return self.as_type or to_pa_datatype(type(self.value))

    def _uuid_keys(self) -> List[Any]:
        return [self.value]