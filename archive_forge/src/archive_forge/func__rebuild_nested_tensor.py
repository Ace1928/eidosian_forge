import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _rebuild_nested_tensor(buffer, sizes, strides, storage_offsets):
    return torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)