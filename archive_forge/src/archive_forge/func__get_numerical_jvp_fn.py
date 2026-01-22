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
def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):

    def jvp_fn(delta):
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)
    return jvp_fn