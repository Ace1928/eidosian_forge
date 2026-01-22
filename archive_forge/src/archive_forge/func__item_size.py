from abc import ABC, abstractmethod
import queue
import threading
import collections
from dataclasses import dataclass
import os
import dataclasses
import io
import pickle
from typing import List, Union, Dict, cast
import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path
from .metadata import (
from .storage import (
from .planner import (
from .utils import _create_file_view
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch._utils import _get_device_module
def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    for s in item.tensor_data.size:
        size *= s
    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)