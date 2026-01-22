from __future__ import annotations
from itertools import islice
from operator import itemgetter
from threading import Lock
from typing import Any
class timetuple(tuple):
    """Tuple of event clock information.

    Can be used as part of a heap to keep events ordered.

    Arguments:
    ---------
        clock (Optional[int]):  Event clock value.
        timestamp (float): Event UNIX timestamp value.
        id (str): Event host id (e.g. ``hostname:pid``).
        obj (Any): Optional obj to associate with this event.
    """
    __slots__ = ()

    def __new__(cls, clock: int | None, timestamp: float, id: str, obj: Any=None) -> timetuple:
        return tuple.__new__(cls, (clock, timestamp, id, obj))

    def __repr__(self) -> str:
        return R_CLOCK.format(*self)

    def __getnewargs__(self) -> tuple:
        return tuple(self)

    def __lt__(self, other: tuple) -> bool:
        try:
            A, B = (self[0], other[0])
            if A and B:
                if A == B:
                    return self[2] < other[2]
                return A < B
            return self[1] < other[1]
        except IndexError:
            return NotImplemented

    def __gt__(self, other: tuple) -> bool:
        return other < self

    def __le__(self, other: tuple) -> bool:
        return not other < self

    def __ge__(self, other: tuple) -> bool:
        return not self < other
    clock = property(itemgetter(0))
    timestamp = property(itemgetter(1))
    id = property(itemgetter(2))
    obj = property(itemgetter(3))