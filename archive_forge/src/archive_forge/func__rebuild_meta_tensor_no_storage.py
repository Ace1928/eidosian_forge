import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad):
    return torch.empty_strided(size, stride, dtype=dtype, device='meta', requires_grad=requires_grad)