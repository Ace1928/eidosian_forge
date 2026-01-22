import typing
from typing import Any, Optional
def _index_intersection(a_idx: tuple[slice, ...], b_idx: tuple[slice, ...], shape: tuple[int, ...]) -> Optional[tuple[slice, ...]]:
    assert len(a_idx) == len(b_idx)
    result = tuple((_slice_intersection(a, b, length) for a, b, length in zip(a_idx, b_idx, shape)))
    if None in result:
        return None
    else:
        return typing.cast(tuple[slice, ...], result)