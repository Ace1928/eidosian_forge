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
def _compare_edges(self, value1: Optional[_T], bound1: str, value2: Optional[_T], bound2: str, only_values: bool=False) -> int:
    """Compare two range bounds.

        Return -1, 0 or 1 respectively when `value1` is less than,
        equal to or greater than `value2`.

        When `only_value` is ``True``, do not consider the *inclusivity*
        of the edges, just their values.
        """
    value1_is_lower_bound = bound1 in {'[', '('}
    value2_is_lower_bound = bound2 in {'[', '('}
    if value1 is value2 is None:
        if value1_is_lower_bound == value2_is_lower_bound:
            return 0
        else:
            return -1 if value1_is_lower_bound else 1
    elif value1 is None:
        return -1 if value1_is_lower_bound else 1
    elif value2 is None:
        return 1 if value2_is_lower_bound else -1
    if bound1 == bound2 and value1 == value2:
        return 0
    value1_inc = bound1 in {'[', ']'}
    value2_inc = bound2 in {'[', ']'}
    step = self._get_discrete_step()
    if step is not None:
        if value1_is_lower_bound:
            if not value1_inc:
                value1 += step
                value1_inc = True
        elif value1_inc:
            value1 += step
            value1_inc = False
        if value2_is_lower_bound:
            if not value2_inc:
                value2 += step
                value2_inc = True
        elif value2_inc:
            value2 += step
            value2_inc = False
    if value1 < value2:
        return -1
    elif value1 > value2:
        return 1
    elif only_values:
        return 0
    elif value1_inc and value2_inc:
        return 0
    elif not value1_inc and (not value2_inc):
        if value1_is_lower_bound == value2_is_lower_bound:
            return 0
        else:
            return 1 if value1_is_lower_bound else -1
    elif not value1_inc:
        return 1 if value1_is_lower_bound else -1
    elif not value2_inc:
        return -1 if value2_is_lower_bound else 1
    else:
        return 0