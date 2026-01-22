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
class MultiRange(List[Range[_T]]):
    """Represents a multirange sequence.

    This list subclass is an utility to allow automatic type inference of
    the proper multi-range SQL type depending on the single range values.
    This is useful when operating on literal multi-ranges::

        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import MultiRange, Range

        value = literal(MultiRange([Range(2, 4)]))

        select(tbl).where(tbl.c.value.op("@")(MultiRange([Range(-3, 7)])))

    .. versionadded:: 2.0.26

    .. seealso::

        - :ref:`postgresql_multirange_list_use`.
    """

    @property
    def __sa_type_engine__(self) -> AbstractMultiRange[_T]:
        return AbstractMultiRange()