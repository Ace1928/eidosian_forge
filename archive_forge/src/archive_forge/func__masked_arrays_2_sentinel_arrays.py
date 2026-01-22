import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _masked_arrays_2_sentinel_arrays(samples):
    has_mask = False
    for sample in samples:
        mask = getattr(sample, 'mask', False)
        has_mask = has_mask or np.any(mask)
    if not has_mask:
        return (samples, None)
    dtype = np.result_type(*samples)
    dtype = dtype if np.issubdtype(dtype, np.number) else np.float64
    for i in range(len(samples)):
        samples[i] = samples[i].astype(dtype, copy=False)
    inexact = np.issubdtype(dtype, np.inexact)
    info = np.finfo if inexact else np.iinfo
    max_possible, min_possible = (info(dtype).max, info(dtype).min)
    nextafter = np.nextafter if inexact else lambda x, _: x - 1
    sentinel = max_possible
    while sentinel > min_possible:
        for sample in samples:
            if np.any(sample == sentinel):
                sentinel = nextafter(sentinel, -np.inf)
                break
        else:
            break
    else:
        message = 'This function replaces masked elements with sentinel values, but the data contains all distinct values of this data type. Consider promoting the dtype to `np.float64`.'
        raise ValueError(message)
    out_samples = []
    for sample in samples:
        mask = getattr(sample, 'mask', None)
        if mask is not None:
            mask = np.broadcast_to(mask, sample.shape)
            sample = sample.data.copy() if np.any(mask) else sample.data
            sample = np.asarray(sample)
            sample[mask] = sentinel
        out_samples.append(sample)
    return (out_samples, sentinel)