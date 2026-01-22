import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _default_to_fused_or_foreach,
from typing import List, Optional
def _multi_tensor_sgd(params: List[Tensor], grads: List[Tensor], momentum_buffer_list: List[Optional[Tensor]], *, weight_decay: float, momentum: float, lr: float, dampening: float, nesterov: bool, maximize: bool, has_sparse_grad: bool):
    if len(params) == 0:
        return
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, momentum_buffer_list], with_indices=True)
    for (device_params, device_grads, device_momentum_buffer_list), indices in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any((grad.is_sparse for grad in device_grads))
        if maximize:
            device_grads = torch._foreach_neg(device_grads)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(device_grads, device_params, alpha=weight_decay)
        if momentum != 0:
            bufs = []
            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])
            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[indices[i]] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)
                    bufs.append(buf)
            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs
        if not device_has_sparse_grad:
            torch._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)