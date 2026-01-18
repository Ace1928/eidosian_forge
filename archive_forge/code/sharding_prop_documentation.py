from functools import lru_cache
from itertools import chain
from typing import Callable, cast, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch._ops import OpOverload
from torch._subclasses import FakeTensorMode
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import TensorMeta
from torch.distributed.device_mesh import DeviceMesh

        Propagate the sharding for an operator given the op_schema.
        