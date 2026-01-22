from __future__ import annotations
import logging # isort:skip
from math import inf
from ..core.enums import Direction
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
class CumSum(Expression):
    """ An expression for generating arrays by cumulatively summing a single
    column from a ``ColumnDataSource``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    field = Required(String, help='\n    The name of a ``ColumnDataSource`` column to cumulatively sum for new values.\n    ')
    include_zero = Bool(default=False, help="\n    Whether to include zero at the start of the result. Note that the length\n    of the result is always the same as the input column. Therefore if this\n    property is True, then the last value of the column will not be included\n    in the sum.\n\n    .. code-block:: python\n\n        source = ColumnDataSource(data=dict(foo=[1, 2, 3, 4]))\n\n        CumSum(field='foo')\n        # -> [1, 3, 6, 10]\n\n        CumSum(field='foo', include_zero=True)\n        # -> [0, 1, 3, 6]\n\n    ")