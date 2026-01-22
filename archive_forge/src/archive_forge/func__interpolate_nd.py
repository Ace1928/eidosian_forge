from __future__ import annotations
from typing import Any, Callable
import numpy as np
from onnx.reference.op_run import OpRun
def _interpolate_nd(data: np.ndarray, get_coeffs: Callable[[float, float], np.ndarray], output_size: list[int] | None=None, scale_factors: list[float] | None=None, axes: list[int] | None=None, roi: np.ndarray | None=None, keep_aspect_ratio_policy: str | None='stretch', exclude_outside: bool=False, **kwargs: Any) -> np.ndarray:
    if output_size is None and scale_factors is None:
        raise ValueError('output_size is None and scale_factors is None.')
    r = len(data.shape)
    if axes is not None:
        if scale_factors is not None:
            new_scale_factors = [1.0] * r
            for i, d in enumerate(axes):
                new_scale_factors[d] = scale_factors[i]
            scale_factors = new_scale_factors
        if output_size is not None:
            new_output_size = [data.shape[i] for i in range(r)]
            for i, d in enumerate(axes):
                new_output_size[d] = output_size[i]
            output_size = new_output_size
        if roi is not None:
            new_roi = [0.0] * r + [1.0] * r
            naxes = len(axes)
            for i, d in enumerate(axes):
                new_roi[d] = roi[i]
                new_roi[r + d] = roi[naxes + i]
            roi = new_roi
    else:
        axes = list(range(r))
    if output_size is not None:
        scale_factors = [output_size[i] / data.shape[i] for i in range(r)]
        if keep_aspect_ratio_policy != 'stretch':
            if keep_aspect_ratio_policy == 'not_larger':
                scale = np.array(scale_factors)[axes].min()
            elif keep_aspect_ratio_policy == 'not_smaller':
                scale = np.array(scale_factors)[axes].max()
            else:
                raise ValueError(f'Invalid keep_aspect_ratio_policy={keep_aspect_ratio_policy!r}')
            scale_factors = [scale if i in axes else 1.0 for i in range(r)]

            def round_half_up(x: float) -> int:
                return int(x + 0.5)
            output_size = [round_half_up(scale * data.shape[i]) if i in axes else data.shape[i] for i in range(r)]
    else:
        output_size = (scale_factors * np.array(data.shape)).astype(int)
    if scale_factors is None:
        raise ValueError('scale_factors is None.')
    if output_size is None:
        raise ValueError('output_size is None.')
    ret = np.zeros(output_size)
    for x in _get_all_coords(ret):
        ret[tuple(x)] = _interpolate_nd_with_x(data, len(data.shape), scale_factors, output_size, x, get_coeffs, roi=roi, exclude_outside=exclude_outside, **kwargs)
    return ret