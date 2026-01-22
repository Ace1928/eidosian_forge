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
class UnaryUfuncInfo(OpInfo):
    """Operator information for 'universal unary functions (unary ufuncs).'
    These are functions of a single tensor with common properties like:
      - they are elementwise functions
      - the input shape is the output shape
      - they typically have method and inplace variants
      - they typically support the out kwarg
      - they typically have NumPy or SciPy references
    See NumPy's universal function documentation
    (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
    about the concept of ufuncs.
    """

    def __init__(self, name, *, dtypes=floating_types(), domain=(None, None), handles_complex_extremal_values=True, handles_large_floats=True, supports_complex_to_float=False, sample_inputs_func=sample_inputs_elementwise_unary, reference_inputs_func=reference_inputs_elementwise_unary, sample_kwargs=lambda device, dtype, input: ({}, {}), reference_numerics_filter=None, **kwargs):
        self._original_unary_ufunc_args = locals().copy()
        super().__init__(name, dtypes=dtypes, sample_inputs_func=sample_inputs_func, reference_inputs_func=reference_inputs_func, **kwargs)
        self.domain = domain
        self.handles_complex_extremal_values = handles_complex_extremal_values
        self.handles_large_floats = handles_large_floats
        self.supports_complex_to_float = supports_complex_to_float
        self.reference_numerics_filter = reference_numerics_filter
        self.sample_kwargs = sample_kwargs
        self._domain_eps = 1e-05