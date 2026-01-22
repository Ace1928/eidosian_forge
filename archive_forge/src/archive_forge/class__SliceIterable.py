from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
class _SliceIterable(EmptyAwareIterable[T]):

    def __init__(self, it: Union[Iterable[T], Iterator[T]], slicer: Any):
        self._n = 0
        self._slicer = slicer
        super().__init__(it)

    def recycle(self) -> None:
        if self._state < 2:
            for _ in self:
                pass
        if self._state == 2:
            self._state = 1

    def _fill_last(self) -> int:
        try:
            if self._state == 0:
                last = self._last
                self._last = next(self._iter)
                is_boundary = self._n > 0 and self._slicer(self._n, self._last, last)
                self._n += 1
                self._state = 2 if is_boundary else 1
        except StopIteration:
            self._state = 3
        return self._state