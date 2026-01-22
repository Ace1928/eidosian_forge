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
def _gradcheck_helper(func, inputs, eps, atol, rtol, nondet_tol, check_undefined_grad, check_grad_dtypes, check_batched_grad, check_batched_forward_grad, check_forward_ad, check_backward_ad, fast_mode, masked):
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs)
    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)
    gradcheck_fn = functools.partial(_fast_gradcheck if fast_mode else _slow_gradcheck, masked=masked)
    _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, check_forward_ad=check_forward_ad, check_backward_ad=check_backward_ad, nondet_tol=nondet_tol, check_undefined_grad=check_undefined_grad)
    if check_batched_forward_grad:
        _test_batched_grad_forward_ad(func, tupled_inputs)
    if not check_backward_ad:
        return True
    for i, o in enumerate(outputs):
        if check_batched_grad:
            _test_batched_grad(tupled_inputs, o, i)
    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)
    if check_undefined_grad and check_backward_ad:
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    return True