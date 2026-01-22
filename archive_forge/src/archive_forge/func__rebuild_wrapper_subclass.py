import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_wrapper_subclass(cls, dtype, size, stride, storage_offset, layout, device, requires_grad):
    return torch.Tensor._make_wrapper_subclass(cls, size, strides=stride, storage_offset=storage_offset, layout=layout, device=device, requires_grad=requires_grad)