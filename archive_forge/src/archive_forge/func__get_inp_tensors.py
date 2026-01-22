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
def _get_inp_tensors(tupled_inputs):
    inp_idx_tup = [(i, t) for i, t in enumerate(tupled_inputs) if is_tensor_like(t) and t.requires_grad]
    return ([tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup])