from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
class Slicer:
    """A better version of :func:`~triad.iter.slice_iterable`

    :param sizer: the function to get size of an item
    :param row_limit: max row for each slice, defaults to None
    :param size_limit: max byte size for each slice, defaults to None
    :param slicer: taking in current number, current value, last value,
        it decides if it's a new slice

    :raises AssertionError: if `size_limit` is not None but `sizer` is None
    """

    def __init__(self, sizer: Optional[Callable[[Any], int]]=None, row_limit: Optional[int]=None, size_limit: Any=None, slicer: Optional[Callable[[int, T, Optional[T]], bool]]=None) -> None:
        self._sizer = sizer
        self._slicer = slicer
        if row_limit is None:
            self._row_limit = 0
        else:
            self._row_limit = row_limit
        if size_limit is None:
            self._size_limit = 0
        else:
            self._size_limit = to_size(str(size_limit))
        assert self._size_limit == 0 or self._sizer is not None, 'sizer must be set when size_limit>0'
        self._current_row = 1
        self._current_size = 0

    def slice(self, orig_it: Iterable[T]) -> Iterable[EmptyAwareIterable[T]]:
        """Slice the original iterable into slices by the combined slicing logic

        :param orig_it: ther original iterable

        :yield: an iterable of EmptyAwareIterable
        """
        it = make_empty_aware(orig_it)
        if it.empty:
            pass
        elif self._row_limit <= 0 and self._size_limit <= 0:
            if self._slicer is None:
                yield it
            else:
                for _slice in slice_iterable(it, self._slicer):
                    yield _slice
        elif self._row_limit > 0 and self._size_limit <= 0:
            if self._slicer is None:
                for _slice in slice_iterable(it, self._is_boundary_row_only):
                    yield _slice
            else:
                for _slice in slice_iterable(it, self._is_boundary_row_only_w_slicer):
                    yield _slice
        else:
            self._current_size = self._sizer(it.peek())
            self._current_row = 1
            if self._row_limit <= 0 and self._size_limit > 0:
                for _slice in slice_iterable(it, self._is_boundary_size_only):
                    yield _slice
            else:
                for _slice in slice_iterable(it, self._is_boundary):
                    yield _slice

    def _is_boundary_row_only(self, no: int, current: Any, last: Any) -> bool:
        return no % self._row_limit == 0

    def _is_boundary_row_only_w_slicer(self, no: int, current: Any, last: Any) -> bool:
        is_boundary = self._slicer is not None and self._slicer(no, current, last)
        if self._current_row >= self._row_limit or is_boundary:
            self._current_row = 1
            return True
        self._current_row += 1
        return False

    def _is_boundary_size_only(self, no: int, current: Any, last: Any) -> bool:
        obj_size = self._sizer(current)
        next_size = self._current_size + obj_size
        is_boundary = self._slicer is not None and self._slicer(no, current, last)
        if next_size > self._size_limit or is_boundary:
            self._current_size = obj_size
            return True
        self._current_size = next_size
        return False

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