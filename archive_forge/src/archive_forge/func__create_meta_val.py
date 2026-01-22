import collections
import itertools
import logging
import operator
import tempfile
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import (
import torch
import torch.fx as fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed._spmd.graph_utils import (
from torch.distributed._spmd.iter_graph_module import IterGraphModule
from torch.fx.passes.shape_prop import TensorMetadata
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _create_meta_val(fake_tensor_mode: FakeTensorMode, val: FakeTensor) -> FakeTensor:
    return FakeTensor(fake_tensor_mode, torch.empty(val.shape, dtype=val.dtype, device='meta', requires_grad=val.requires_grad), val.device)