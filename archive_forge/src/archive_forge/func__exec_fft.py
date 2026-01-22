import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _exec_fft(a, direction, value_type, norm, axis, overwrite_x, out_size=None, out=None, plan=None):
    fft_type = _convert_fft_type(a.dtype, value_type)
    if axis % a.ndim != a.ndim - 1:
        a = a.swapaxes(axis, -1)
    if a.base is not None or not a.flags.c_contiguous:
        a = a.copy()
    elif not cupy.cuda.runtime.is_hip and value_type == 'C2R' and (not overwrite_x) and (10010 <= cupy.cuda.runtime.runtimeGetVersion()):
        a = a.copy()
    elif cupy.cuda.runtime.is_hip and value_type != 'C2C':
        a = a.copy()
    n = a.shape[-1]
    if n < 1:
        raise ValueError('Invalid number of FFT data points (%d) specified.' % n)
    if cupy.cuda.runtime.is_hip and value_type == 'C2R':
        a[..., 0].imag = 0
        if out_size is None:
            a[..., -1].imag = 0
        elif out_size % 2 == 0:
            a[..., out_size // 2].imag = 0
    if out_size is None:
        out_size = n
    batch = a.size // n
    curr_plan = cufft.get_current_plan()
    if curr_plan is not None:
        if plan is None:
            plan = curr_plan
        else:
            raise RuntimeError('Use the cuFFT plan either as a context manager or as an argument.')
    if plan is None:
        devices = None if not config.use_multi_gpus else config._devices
        keys = (out_size, fft_type, batch, devices)
        mgr = config.get_current_callback_manager()
        if mgr is not None:
            load_aux = mgr.cb_load_aux_arr
            store_aux = mgr.cb_store_aux_arr
            keys += (mgr.cb_load, mgr.cb_store, 0 if load_aux is None else load_aux.data.ptr, 0 if store_aux is None else store_aux.data.ptr)
        cache = get_plan_cache()
        cached_plan = cache.get(keys)
        if cached_plan is not None:
            plan = cached_plan
        elif mgr is None:
            plan = cufft.Plan1d(out_size, fft_type, batch, devices=devices)
            cache[keys] = plan
        else:
            if devices:
                raise NotImplementedError('multi-GPU cuFFT callbacks are not yet supported')
            plan = mgr.create_plan(('Plan1d', keys[:-5]))
            mgr.set_callbacks(plan)
            cache[keys] = plan
    else:
        if not isinstance(plan, cufft.Plan1d):
            raise ValueError('expected plan to have type cufft.Plan1d')
        if fft_type != plan.fft_type:
            raise ValueError('cuFFT plan dtype mismatch.')
        if out_size != plan.nx:
            raise ValueError('Target array size does not match the plan.', out_size, plan.nx)
        if batch != plan.batch:
            raise ValueError('Batch size does not match the plan.')
        if config.use_multi_gpus != (plan.gpus is not None):
            raise ValueError('Unclear if multiple GPUs are to be used or not.')
    if overwrite_x and value_type == 'C2C':
        out = a
    elif out is not None:
        plan.check_output_array(a, out)
    else:
        out = plan.get_output_array(a)
    if batch != 0:
        plan.fft(a, out, direction)
    sz = out.shape[-1]
    if fft_type == cufft.CUFFT_R2C or fft_type == cufft.CUFFT_D2Z:
        sz = n
    if norm == 'backward' and direction == cufft.CUFFT_INVERSE:
        out /= sz
    elif norm == 'ortho':
        out /= math.sqrt(sz)
    elif norm == 'forward' and direction == cufft.CUFFT_FORWARD:
        out /= sz
    if axis % a.ndim != a.ndim - 1:
        out = out.swapaxes(axis, -1)
    return out