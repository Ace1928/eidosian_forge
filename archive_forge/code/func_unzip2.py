from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload
def unzip2(xys: Iterable[tuple[T, S]]) -> tuple[tuple[T, ...], tuple[S, ...]]:
    """Unzip sequence of length-2 tuples into two tuples."""
    xs = []
    ys = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return (tuple(xs), tuple(ys))