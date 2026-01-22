import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _stack_if_compiling,
from typing import List, Optional
def _single_tensor_adamax(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_infs: List[Tensor], state_steps: List[Tensor], *, eps: float, beta1: float, beta2: float, lr: float, weight_decay: float, maximize: bool, differentiable: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad
        exp_avg = exp_avgs[i]
        exp_inf = exp_infs[i]
        step_t = state_steps[i]
        step_t += 1
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_inf = torch.view_as_real(exp_inf)
        exp_avg.lerp_(grad, 1 - beta1)
        norm_buf = torch.cat([exp_inf.mul_(beta2).unsqueeze(0), grad.abs().add_(eps).unsqueeze_(0)], 0)
        if not differentiable:
            torch.amax(norm_buf, 0, keepdim=False, out=exp_inf)
        else:
            exp_inf.copy_(torch.amax(norm_buf, 0, keepdim=False))
        bias_correction = 1 - beta1 ** _get_value(step_t)
        clr = lr / bias_correction
        param.addcdiv_(exp_avg, exp_inf, value=-clr)