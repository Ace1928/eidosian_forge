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
def unwrap_to_op_info(self, op_call: torch._ops.OpOverload, args: Tuple[object, ...], kwargs: Dict[str, object]) -> OpInfo:
    runtime_schema_info = self.sharding_propagator.op_to_schema_info.get(op_call, None)
    if runtime_schema_info is not None and runtime_schema_info.needs_pytree:
        tree_args, args_spec = pytree.tree_flatten(args)
        args_list: Sequence[object] = tree_args
    else:
        args_list, args_spec = (args, None)
    args_schema: List[object] = []
    kwargs_schema: Dict[str, object] = {}
    local_args: List[object] = []
    local_kwargs: Dict[str, object] = {}
    mesh: Optional[DeviceMesh] = None
    for arg in args_list:
        if isinstance(arg, dtensor.DTensor):
            args_schema.append(arg._spec)
            local_args.append(arg._local_tensor)
            if mesh is not None:
                if mesh != arg.device_mesh:
                    raise NotImplementedError(f'{op_call}: DTensor does not support cross-mesh operation yet!')
            else:
                mesh = arg.device_mesh
        elif isinstance(arg, torch.Tensor):
            if arg.ndim == 0 and mesh is not None:
                args_schema.append(DTensorSpec(mesh, (Replicate(),) * mesh.ndim, tensor_meta=TensorMeta(shape=arg.shape, stride=arg.stride(), dtype=arg.dtype)))
                local_args.append(arg)
            else:
                raise RuntimeError(f'{op_call}: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!')
        else:
            args_schema.append(arg)
            local_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, dtensor.DTensor):
            kwargs_schema[k] = v._spec
            local_kwargs[k] = v._local_tensor
            if mesh is not None:
                if mesh != v.device_mesh:
                    raise NotImplementedError(f'{op_call}: DTensor does not support cross-mesh operation yet!')
            else:
                mesh = v.device_mesh
        elif isinstance(v, torch.Tensor):
            raise RuntimeError(f'{op_call}: got mixed torch.Tensor and DTensor, need to convert all torch.Tensor to DTensor before calling distributed operators!')
        else:
            kwargs_schema[k] = v
            local_kwargs[k] = v
    assert mesh is not None, f'found no DeviceMesh from dtensor args for {op_call}!'
    op_info = OpInfo(mesh, OpSchema(op_call, pytree.tree_unflatten(args_schema, args_spec) if args_spec else tuple(args_schema), kwargs_schema, schema_info=runtime_schema_info), args_schema, tuple(local_args), local_kwargs, args_spec)
    return op_info