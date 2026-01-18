import inspect
from typing import Callable, List, Optional, Set, Tuple, Union
import torch
from packaging import version
from safetensors.torch import storage_ptr, storage_size
from torch import nn
from .utils import is_torch_tpu_available, logging

    Unique identifier to a tensor storage. Multiple different tensors can share the same underlying storage. For
    example, "meta" tensors all share the same storage, and thus their identifier will all be equal. This identifier is
    guaranteed to be unique and constant for this tensor's storage during its lifetime. Two tensor storages with
    non-overlapping lifetimes may have the same id.
    