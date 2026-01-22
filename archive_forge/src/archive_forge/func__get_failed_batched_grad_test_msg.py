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
def _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp, is_forward_ad=False):
    return f'\nFor output {output_idx} and input {input_idx}:\n\n{(FAILED_BATCHED_GRAD_MSG_FWD_AD if is_forward_ad else FAILED_BATCHED_GRAD_MSG)}\n\nGot:\n{res}\n\nExpected:\n{exp}\n'.strip()