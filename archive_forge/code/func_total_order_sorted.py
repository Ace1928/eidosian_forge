from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, overload
def total_order_sorted(iterable: Iterable[T], *, key: Callable[[T], Any] | None=None, reverse: bool=False) -> list[T]:
    """Sort an iterable in a total order.

    This is useful for sorting objects that are not comparable, e.g., dictionaries with different
    types of keys.
    """
    sequence = list(iterable)
    try:
        return sorted(sequence, key=key, reverse=reverse)
    except TypeError:
        if key is None:

            def key_fn(x: T) -> tuple[str, Any]:
                return (f'{x.__class__.__module__}.{x.__class__.__qualname__}', x)
        else:

            def key_fn(x: T) -> tuple[str, Any]:
                y = key(x)
                return (f'{y.__class__.__module__}.{y.__class__.__qualname__}', y)
        try:
            return sorted(sequence, key=key_fn, reverse=reverse)
        except TypeError:
            return sequence