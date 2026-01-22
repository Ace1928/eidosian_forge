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
def _upper_edge_adjacent_to_lower(self, value1: Optional[_T], bound1: str, value2: Optional[_T], bound2: str) -> bool:
    """Determine whether an upper bound is immediately successive to a
        lower bound."""
    res = self._compare_edges(value1, bound1, value2, bound2, True)
    if res == -1:
        step = self._get_discrete_step()
        if step is None:
            return False
        if bound1 == ']':
            if bound2 == '[':
                return value1 == value2 - step
            else:
                return value1 == value2
        elif bound2 == '[':
            return value1 == value2
        else:
            return value1 == value2 - step
    elif res == 0:
        if bound1 == ']' and bound2 == '[' or (bound1 == ')' and bound2 == '('):
            step = self._get_discrete_step()
            if step is not None:
                return True
        return bound1 == ')' and bound2 == '[' or (bound1 == ']' and bound2 == '(')
    else:
        return False