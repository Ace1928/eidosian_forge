import functools
import operator
from typing import cast, Dict, List, Optional, Sequence, Tuple
import torch
import torch.distributed as dist
import torch.distributed._tensor.api as dtensor
import torch.distributed._tensor.random as random
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.placement_types import DTensorSpec, Replicate, TensorMeta
from torch.distributed._tensor.random import is_rng_supported_mesh
from torch.distributed._tensor.redistribute import redistribute_local_tensor
from torch.distributed._tensor.sharding_prop import ShardingPropagator
from torch.distributed._tensor.tp_conv import (
from torch.distributed.device_mesh import DeviceMesh
def to_dt(res, spec):
    assert spec is not None and isinstance(spec, DTensorSpec), f'output spec does not match with output! Expected DTensorSpec, got {spec}.'
    assert spec.tensor_meta is not None
    return dtensor.DTensor(res, spec.mesh, spec.placements, shape=spec.tensor_meta.shape, dtype=spec.tensor_meta.dtype, requires_grad=res.requires_grad, stride=spec.tensor_meta.stride)