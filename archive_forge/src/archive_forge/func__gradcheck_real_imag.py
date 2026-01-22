import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
def _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, check_forward_ad, check_backward_ad, nondet_tol, check_undefined_grad):
    complex_out_indices = [i for i, o in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any((o.is_complex() for o in _as_tuple(func_out)))
    if check_backward_ad:
        if has_any_complex_output:
            real_fn, imag_fn = _real_and_imag_output(func)
            imag_func_out = imag_fn(*tupled_inputs)
            imag_outputs = _differentiable_outputs(imag_func_out)
            gradcheck_fn(imag_fn, imag_func_out, tupled_inputs, imag_outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_out_indices, test_imag=True)
            real_func_out = real_fn(*tupled_inputs)
            real_outputs = _differentiable_outputs(real_func_out)
            gradcheck_fn(real_fn, real_func_out, tupled_inputs, real_outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_out_indices)
        else:
            gradcheck_fn(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol)
    if check_forward_ad:
        complex_inp_indices = [i for i, inp in enumerate(tupled_inputs) if is_tensor_like(inp) and inp.is_complex()]
        if complex_inp_indices:
            real_fn, imag_fn = _real_and_imag_input(func, complex_inp_indices, tupled_inputs)
            imag_inputs = [inp.imag if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(imag_fn, imag_func_out, imag_inputs, diff_imag_func_out, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_inp_indices, test_imag=True, use_forward_ad=True)
            real_inputs = [inp.real if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(real_fn, real_func_out, real_inputs, diff_real_func_out, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_inp_indices, use_forward_ad=True)
            if check_undefined_grad:
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        else:
            gradcheck_fn(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, use_forward_ad=True)
            if check_undefined_grad:
                _test_undefined_forward_mode(func, outputs, tupled_inputs)