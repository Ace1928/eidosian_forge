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
class comparator_factory(TypeEngine.Comparator[Range[Any]]):
    """Define comparison operations for range types."""

    def contains(self, other: Any, **kw: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the right hand operand,
            which can be an element or a range, is contained within the
            column.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
        return self.expr.operate(CONTAINS, other)

    def contained_by(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the column is contained
            within the right hand operand.
            """
        return self.expr.operate(CONTAINED_BY, other)

    def overlaps(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the column overlaps
            (has points in common with) the right hand operand.
            """
        return self.expr.operate(OVERLAP, other)

    def strictly_left_of(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the column is strictly
            left of the right hand operand.
            """
        return self.expr.operate(STRICTLY_LEFT_OF, other)
    __lshift__ = strictly_left_of

    def strictly_right_of(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the column is strictly
            right of the right hand operand.
            """
        return self.expr.operate(STRICTLY_RIGHT_OF, other)
    __rshift__ = strictly_right_of

    def not_extend_right_of(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the range in the column
            does not extend right of the range in the operand.
            """
        return self.expr.operate(NOT_EXTEND_RIGHT_OF, other)

    def not_extend_left_of(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the range in the column
            does not extend left of the range in the operand.
            """
        return self.expr.operate(NOT_EXTEND_LEFT_OF, other)

    def adjacent_to(self, other: Any) -> ColumnElement[bool]:
        """Boolean expression. Returns true if the range in the column
            is adjacent to the range in the operand.
            """
        return self.expr.operate(ADJACENT_TO, other)

    def union(self, other: Any) -> ColumnElement[bool]:
        """Range expression. Returns the union of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
        return self.expr.operate(operators.add, other)

    def difference(self, other: Any) -> ColumnElement[bool]:
        """Range expression. Returns the union of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
        return self.expr.operate(operators.sub, other)

    def intersection(self, other: Any) -> ColumnElement[Range[_T]]:
        """Range expression. Returns the intersection of the two ranges.
            Will raise an exception if the resulting range is not
            contiguous.
            """
        return self.expr.operate(operators.mul, other)