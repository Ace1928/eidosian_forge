import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
class ReductionOpInfo(OpInfo):
    """Reduction operator information.

    An operator is a reduction operator if it reduces one or more dimensions of
    the input tensor to a single value. Reduction operators must implement the
    following signature:

    - `op(input, *args, *, dim=None, keepdim=False, **kwargs) -> Tensor`

    ReductionOpInfo tests that reduction operators implement a consistent API.
    Optional features such as reducing over multiple dimensions are captured in
    the optional keyword parameters of the ReductionOpInfo constructor.

    If a reduction operator does not yet implement the full required API of
    reduction operators, this should be documented by xfailing the failing
    tests rather than adding optional parameters to ReductionOpInfo.

    NOTE
    The API for reduction operators has not yet been finalized and some
    requirements may change.

    See tests in test/test_reductions.py
    """

    def __init__(self, name, *, identity: Optional[Any]=None, nan_policy: Optional[str]=None, supports_multiple_dims: bool=True, promotes_int_to_float: bool=False, promotes_int_to_int64: bool=False, result_dtype: Optional[torch.dtype]=None, complex_to_real: bool=False, generate_args_kwargs: Callable=lambda t, dim=None, keepdim=False: (yield (tuple(), {})), **kwargs):
        self._original_reduction_args = locals().copy()
        assert nan_policy in (None, 'propagate', 'omit')
        assert not (result_dtype and promotes_int_to_float)
        assert not (result_dtype and promotes_int_to_int64)
        assert not (result_dtype and complex_to_real)
        assert not (promotes_int_to_float and promotes_int_to_int64)

        def sample_inputs_func(*args, **kwargs):
            kwargs['supports_multiple_dims'] = supports_multiple_dims
            kwargs['generate_args_kwargs'] = generate_args_kwargs
            yield from sample_inputs_reduction(*args, **kwargs)
        kwargs.setdefault('inplace_variant', None)
        kwargs.setdefault('sample_inputs_func', sample_inputs_func)
        super().__init__(name, promotes_int_to_float=promotes_int_to_float, **kwargs)
        self.identity = identity
        self.nan_policy = nan_policy
        self.supports_multiple_dims = supports_multiple_dims
        self.promotes_int_to_int64 = promotes_int_to_int64
        self.complex_to_real = complex_to_real
        self.result_dtype = result_dtype
        self.generate_args_kwargs = generate_args_kwargs