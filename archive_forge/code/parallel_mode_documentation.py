from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.distributed._spmd.data_parallel import (
from torch.distributed._spmd.distribute import _convert_to_distributed, Schema
from torch.distributed._tensor import DeviceMesh, Placement, Replicate, Shard
from torch.fx import GraphModule

        Transform and compile a distributed graph with a set of graph transformation
        and optimization passes for the dtensor fallback parallel mode.
        