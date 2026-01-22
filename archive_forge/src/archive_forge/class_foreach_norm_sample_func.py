from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
class foreach_norm_sample_func(foreach_inputs_sample_func):

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for ord in (0, 1, 2, -1, -2):
            input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs)
            disable_fastpath = True
            if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
                disable_fastpath = False
            yield ForeachSampleInput(input, ord=ord, disable_fastpath=disable_fastpath)

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        num_input_tensors = kwargs.pop('num_input_tensors', foreach_num_tensors)
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for k, v in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for num_tensors, ord in product(num_input_tensors, (0, 1, 2, -1, -2)):
            input = sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            disable_fastpath = True
            if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
                disable_fastpath = False
            yield ForeachSampleInput(input, ord=ord, disable_fastpath=disable_fastpath)