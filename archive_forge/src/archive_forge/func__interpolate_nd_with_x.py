from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _interpolate_nd_with_x(data: np.ndarray, n: int, scale_factors: list[float], output_size: list[int], x: list[float], get_coeffs: Callable[[float, float], np.ndarray], roi: np.ndarray | None=None, exclude_outside: bool=False, **kwargs: Any) -> np.ndarray:
    if n == 1:
        return _interpolate_1d_with_x(data, scale_factors[0], output_size[0], x[0], get_coeffs, roi=roi, exclude_outside=exclude_outside, **kwargs)
    res1d = []
    for i in range(data.shape[0]):
        r = _interpolate_nd_with_x(data[i], n - 1, scale_factors[1:], output_size[1:], x[1:], get_coeffs, roi=None if roi is None else np.concatenate([roi[1:n], roi[n + 1:]]), exclude_outside=exclude_outside, **kwargs)
        res1d.append(r)
    return _interpolate_1d_with_x(res1d, scale_factors[0], output_size[0], x[0], get_coeffs, roi=None if roi is None else [roi[0], roi[n]], exclude_outside=exclude_outside, **kwargs)