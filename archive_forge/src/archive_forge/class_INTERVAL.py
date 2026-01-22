from __future__ import annotations
import datetime as dt
from typing import Any
from typing import Optional
from typing import overload
from typing import Type
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from ...sql import sqltypes
from ...sql import type_api
from ...util.typing import Literal
class INTERVAL(type_api.NativeForEmulated, sqltypes._AbstractInterval):
    """PostgreSQL INTERVAL type."""
    __visit_name__ = 'INTERVAL'
    native = True

    def __init__(self, precision: Optional[int]=None, fields: Optional[str]=None) -> None:
        """Construct an INTERVAL.

        :param precision: optional integer precision value
        :param fields: string fields specifier.  allows storage of fields
         to be limited, such as ``"YEAR"``, ``"MONTH"``, ``"DAY TO HOUR"``,
         etc.

         .. versionadded:: 1.2

        """
        self.precision = precision
        self.fields = fields

    @classmethod
    def adapt_emulated_to_native(cls, interval: sqltypes.Interval, **kw: Any) -> INTERVAL:
        return INTERVAL(precision=interval.second_precision)

    @property
    def _type_affinity(self) -> Type[sqltypes.Interval]:
        return sqltypes.Interval

    def as_generic(self, allow_nulltype: bool=False) -> sqltypes.Interval:
        return sqltypes.Interval(native=True, second_precision=self.precision)

    @property
    def python_type(self) -> Type[dt.timedelta]:
        return dt.timedelta

    def literal_processor(self, dialect: Dialect) -> Optional[_LiteralProcessorType[dt.timedelta]]:

        def process(value: dt.timedelta) -> str:
            return f'make_interval(secs=>{value.total_seconds()})'
        return process