import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
def _to_will_alias(a: TensorLikeType, device: Optional[DeviceLikeType]=None, dtype: Optional[torch.dtype]=None, copy: Optional[bool]=None, layout: Optional[torch.layout]=None, memory_format: Optional[torch.memory_format]=None, pin_memory: Optional[bool]=False, non_blocking: bool=False) -> bool:
    return not copy and (device is None or a.device == device) and (dtype is None or a.dtype == dtype) and (layout is None or a.layout == layout) and (memory_format is None or memory_format == torch.preserve_format or utils.is_contiguous_for_memory_format(a, memory_format=memory_format))