from __future__ import annotations
import dataclasses
from datetime import date
from datetime import datetime
from datetime import timedelta
from decimal import Decimal
from typing import Any
from typing import cast
from typing import Generic
from typing import List
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .operators import ADJACENT_TO
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import NOT_EXTEND_LEFT_OF
from .operators import NOT_EXTEND_RIGHT_OF
from .operators import OVERLAP
from .operators import STRICTLY_LEFT_OF
from .operators import STRICTLY_RIGHT_OF
from ... import types as sqltypes
from ...sql import operators
from ...sql.type_api import TypeEngine
from ...util import py310
from ...util.typing import Literal
class AbstractMultiRange(AbstractRange[Sequence[Range[_T]]]):
    """Base for PostgreSQL MULTIRANGE types.

    these are types that return a sequence of :class:`_postgresql.Range`
    objects.

    """
    __abstract__ = True

    def _resolve_for_literal(self, value: Sequence[Range[Any]]) -> Any:
        if not value:
            return sqltypes.NULLTYPE
        first = value[0]
        spec = first.lower if first.lower is not None else first.upper
        if isinstance(spec, int):
            if all((_is_int32(r) for r in value)):
                return INT4MULTIRANGE()
            else:
                return INT8MULTIRANGE()
        elif isinstance(spec, (Decimal, float)):
            return NUMMULTIRANGE()
        elif isinstance(spec, datetime):
            return TSMULTIRANGE() if not spec.tzinfo else TSTZMULTIRANGE()
        elif isinstance(spec, date):
            return DATEMULTIRANGE()
        else:
            return sqltypes.NULLTYPE