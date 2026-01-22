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
def _check_analytical_numerical_equal(all_analytical, all_numerical, complex_indices, tupled_inputs, outputs, func, all_v, all_u, rtol, atol, test_imag, *, is_forward_ad=False):
    for i, all_numerical_for_input_i in enumerate(all_numerical):
        for j, n in enumerate(all_numerical_for_input_i):
            if is_forward_ad:
                a = all_analytical[i][j]
            else:
                a = all_analytical[j][i]
            n = n.to(device=a.device)
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                jacobians_str = _run_slow_mode_and_get_error(func, tupled_inputs, outputs, i, j, rtol, atol, is_forward_ad)
                raise GradcheckError(_get_notallclose_msg(a, n, j, i, complex_indices, test_imag, is_forward_ad) + jacobians_str)