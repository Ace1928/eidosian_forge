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
def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):

    def apply_to_c_inps(fn, fn_to_apply):

        def wrapped_fn(*inputs):
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(new_inputs[should_be_complex], tupled_inputs[should_be_complex])
            return _as_tuple(fn(*new_inputs))
        return wrapped_fn
    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return (real_fn, imag_fn)