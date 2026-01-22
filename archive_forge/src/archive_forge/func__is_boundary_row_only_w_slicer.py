from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def _is_boundary_row_only_w_slicer(self, no: int, current: Any, last: Any) -> bool:
    is_boundary = self._slicer is not None and self._slicer(no, current, last)
    if self._current_row >= self._row_limit or is_boundary:
        self._current_row = 1
        return True
    self._current_row += 1
    return False