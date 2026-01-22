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
def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    on_index = 'on index {idx} ' if idx is not None else ''
    assert output1.shape == output2.shape, f'Expected `func` to return outputs with the same shape when inputs are perturbed {on_index}by {eps}, but got: shapes {output1.shape} and {output2.shape}.'
    assert output1.dtype == output2.dtype, f'Expected `func` to return outputs with the same dtype when inputs are perturbed {on_index}by {eps}, but got: dtypes {output1.dtype} and {output2.dtype}.'