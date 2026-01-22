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
class _TestParamsMaxPoolBase:

    def __init__(self):
        self.kwargs = {'kernel_size': [3], 'stride': [2, None], 'ceil_mode': [True, False], 'padding': [0, 1], 'dilation': [1], 'return_indices': [True, False]}
        self.shapes = [[1, 2, None], [2], [3, 6]]

    def _gen_shape(self):
        for shape in product(*self.shapes):
            if shape[0] is None:
                shape = shape[1:]
            yield (shape, torch.contiguous_format)
            if len(self.shapes) == 4 and len(shape) == 4:
                yield (shape, torch.channels_last)

    def _gen_kwargs(self):
        keys = self.kwargs.keys()
        for values in product(*self.kwargs.values()):
            yield dict(zip(keys, values))

    def gen_input_params(self):
        yield from product(self._gen_shape(), self._gen_kwargs())