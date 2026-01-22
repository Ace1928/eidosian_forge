from __future__ import annotations
from typing import TYPE_CHECKING, cast
from more_itertools import unique_everseen
def _iter_all_subclasses(cls: type[object]) -> Iterator[type[Any]]:
    try:
        subs = cls.__subclasses__()
    except TypeError:
        subs = cast('type[type]', cls).__subclasses__(cls)
    for sub in subs:
        yield sub
        yield from iter_subclasses(sub)