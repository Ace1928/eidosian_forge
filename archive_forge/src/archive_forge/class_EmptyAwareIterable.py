from typing import (
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_size
class EmptyAwareIterable(Iterable[T]):
    """A wrapper of iterable that can tell if the underlying
    iterable is empty, it can also peek a non-empty iterable.

    :param it: the underlying iterable

    :raises StopIteration: raised by the underlying iterable
    """

    def __init__(self, it: Union[Iterable[T], Iterator[T]]):
        self._last: Optional[T] = None
        if not isinstance(it, Iterator):
            self._iter = iter(it)
        else:
            self._iter = it
        self._state = 0
        self._fill_last()

    @property
    def empty(self) -> bool:
        """Check if the underlying iterable has more items

        :return: whether it is empty
        """
        return self._fill_last() >= 2

    def peek(self) -> T:
        """Return the next of the iterable without moving

        :raises StopIteration: if it's empty
        :return: the `next` item
        """
        if not self.empty:
            return self._last
        raise StopIteration("Can't peek empty iterable")

    def __iter__(self) -> Any:
        """Wrapper of the underlying __iter__

        :yield: next object
        """
        while self._fill_last() < 2:
            self._state = 0
            yield self._last

    def _fill_last(self) -> int:
        try:
            if self._state == 0:
                self._last = next(self._iter)
                self._state = 1
        except StopIteration:
            self._state = 3
        return self._state