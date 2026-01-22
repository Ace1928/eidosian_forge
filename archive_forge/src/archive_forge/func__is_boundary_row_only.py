from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def _is_boundary_row_only(self, no: int, current: Any, last: Any) -> bool:
    return no % self._row_limit == 0