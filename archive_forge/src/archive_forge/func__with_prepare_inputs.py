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
def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):

    def wrapped_fn():
        inp = tuple((_prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode) if is_tensor_like(a) else a for i, a in enumerate(_as_tuple(inputs))))
        return tuple((a.clone() for a in _as_tuple(fn(*inp))))
    return wrapped_fn