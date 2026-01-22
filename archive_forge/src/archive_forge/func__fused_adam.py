from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor
from .optimizer import (Optimizer, ParamsT, _use_grad_for_differentiable, _get_value,
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
def _fused_adam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor], state_steps: List[Tensor], grad_scale: Optional[Tensor], found_inf: Optional[Tensor], *, amsgrad: bool, has_complex: bool, beta1: float, beta2: float, lr: Union[float, Tensor], weight_decay: float, eps: float, maximize: bool, capturable: bool, differentiable: bool) -> None:
    if not params:
        return
    if differentiable:
        raise RuntimeError('Adam with fused=True does not support differentiable=True')
    grad_scale_dict = {grad_scale.device: grad_scale} if grad_scale is not None else None
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None
    lr_dict = {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != 'cpu' else None
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps])
    for (device, _), ((device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps), _) in grouped_tensors.items():
        device_grad_scale, device_found_inf = (None, None)
        if grad_scale is not None:
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)
            device_grad_scale = grad_scale_dict[device]
        if found_inf is not None:
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)
            device_found_inf = found_inf_dict[device]
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]
        torch._foreach_add_(device_state_steps, 1)
        torch._fused_adam_(device_params, device_grads, device_exp_avgs, device_exp_avg_sqs, device_max_exp_avg_sqs, device_state_steps, amsgrad=amsgrad, lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps, maximize=maximize, grad_scale=device_grad_scale, found_inf=device_found_inf)
        if device_found_inf is not None:
            torch._foreach_sub_(device_state_steps, [device_found_inf] * len(device_state_steps))