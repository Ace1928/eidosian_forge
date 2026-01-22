from typing import cast, Dict, List, Tuple
import torch
import torch.distributed._tensor.api as dtensor
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
class Redistribute(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: 'dtensor.DTensor', device_mesh: DeviceMesh, placements: Tuple[Placement, ...]):
        current_spec = input._spec
        ctx.current_spec = current_spec
        target_spec = DTensorSpec(device_mesh, placements, tensor_meta=input._spec.tensor_meta)
        local_tensor = input._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec)
        return dtensor.DTensor(output, device_mesh, target_spec.placements, shape=input.shape, dtype=input.dtype, requires_grad=input.requires_grad, stride=input.stride())

    @staticmethod
    def backward(ctx, grad_output: 'dtensor.DTensor'):
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec
        target_placements: List[Placement] = []
        for current, target in zip(current_spec.placements, previous_spec.placements):
            if not current.is_partial() and target.is_partial():
                target_placements.append(Replicate())
            else:
                target_placements.append(target)
        target_spec = DTensorSpec(previous_spec.mesh, tuple(target_placements), tensor_meta=previous_spec.tensor_meta)
        local_tensor = grad_output._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec)
        output_dtensor = dtensor.DTensor(output, target_spec.mesh, target_spec.placements, shape=grad_output.shape, dtype=grad_output.dtype, requires_grad=grad_output.requires_grad, stride=grad_output.stride())
        return (output_dtensor, None, None)