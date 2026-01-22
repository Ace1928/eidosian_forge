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
def _fast_gradcheck(func, func_out, inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, *, use_forward_ad=False, complex_indices=None, test_imag=False, masked=False):
    inp_tensors_idx, inp_tensors = _get_inp_tensors(inputs)
    all_v, all_u, all_u_dense = _make_vectors(inp_tensors, outputs, use_forward_ad=use_forward_ad)
    inputs_numerical, all_u_numerical, all_v_numerical = (inputs, all_u, all_v) if masked else _densify((inputs, all_u, all_v))
    numerical_vJu = _get_numerical_vJu(func, inputs_numerical, inp_tensors_idx, func_out, all_u_numerical, all_v_numerical, eps, is_forward_ad=use_forward_ad)
    if use_forward_ad:
        assert all_v is None
        analytical_vJu = _get_analytical_jacobian_forward_ad(func, inputs, _as_tuple(func_out), all_u=all_u, check_grad_dtypes=check_grad_dtypes)
    else:
        if not outputs:
            _check_no_differentiable_outputs_fast(func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol)
        analytical_vJu = _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense)
    _check_analytical_numerical_equal(analytical_vJu, numerical_vJu, complex_indices, inputs, outputs, func, all_v, all_u, rtol, atol, test_imag, is_forward_ad=use_forward_ad)
    return True