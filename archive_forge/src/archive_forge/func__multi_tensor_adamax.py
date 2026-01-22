import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
from typing import List, Optional
def _multi_tensor_adamax(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_infs: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, eps: float, maximize: bool, differentiable: bool, has_complex: bool):
    assert not differentiable, "_foreach ops don't support autograd"
    if len(params) == 0:
        return
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, exp_avgs, exp_infs, state_steps])
    for (grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_infs, grouped_state_steps), _ in grouped_tensors.values():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_infs)
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        torch._foreach_mul_(grouped_exp_infs, beta2)
        for exp_inf, grad in zip(grouped_exp_infs, grouped_grads):
            norm_buf = torch.cat([exp_inf.unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0)
            torch.max(norm_buf, 0, keepdim=False, out=(exp_inf, exp_inf.new().long()))
        bias_corrections = [1 - beta1 ** _get_value(step) for step in grouped_state_steps]
        clr = _stack_if_compiling([-1 * (lr / bias_correction) for bias_correction in bias_corrections])
        torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, grouped_exp_infs, clr)