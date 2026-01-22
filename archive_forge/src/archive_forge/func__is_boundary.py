from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
def _is_boundary(self, no: int, current: Any, last: Any) -> bool:
    obj_size = self._sizer(current)
    next_size = self._current_size + obj_size
    is_boundary = self._slicer is not None and self._slicer(no, current, last)
    if next_size > self._size_limit or self._current_row >= self._row_limit or is_boundary:
        self._current_size = obj_size
        self._current_row = 1
        return True
    self._current_size = next_size
    self._current_row += 1
    return False