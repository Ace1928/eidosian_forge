import typing
from typing import Any, Optional
def _slice_intersection(a: slice, b: slice, length: int) -> Optional[slice]:
    a_start, a_stop, a_step = a.indices(length)
    b_start, b_stop, b_step = b.indices(length)
    crt_result = _crt(a_start, a_step, b_start, b_step)
    if crt_result is None:
        return None
    c_start, c_step = crt_result
    c_stop = min(a_stop, b_stop)
    if c_start >= c_stop:
        return None
    return slice(c_start, c_stop, c_step)