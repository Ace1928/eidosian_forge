import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
from torch._utils import is_compiling
from typing import List, Optional
def _multi_tensor_asgd(params: List[Tensor], grads: List[Tensor], axs: List[Tensor], mus: List[Tensor], etas: List[Tensor], state_steps: List[Tensor], *, lambd: float, lr: float, t0: float, alpha: float, weight_decay: float, maximize: bool, differentiable: bool, capturable: bool, has_complex: bool):
    if len(params) == 0:
        return
    assert not differentiable, "_foreach ops don't support autograd"
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype([params, grads, axs, mus, etas, state_steps])
    for (device, _), ((grouped_params, grouped_grads, grouped_axs, grouped_mus, grouped_etas, grouped_state_steps), _) in grouped_tensors.items():
        if maximize:
            grouped_grads = torch._foreach_neg(grouped_grads)
        grouped_grads = list(grouped_grads)
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_axs)
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(grouped_state_steps, torch.tensor(1.0, device='cpu'), alpha=1.0)
        else:
            torch._foreach_add_(grouped_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
                intermediate = grouped_grads
            else:
                intermediate = torch._foreach_add(grouped_grads, grouped_params, alpha=weight_decay)
            torch._foreach_add_(intermediate, grouped_params, alpha=lambd)
        else:
            intermediate = torch._foreach_add(grouped_grads, grouped_params, alpha=lambd)
        torch._foreach_addcmul_(grouped_params, intermediate, grouped_etas, value=-1)
        del intermediate
        intermediate = torch._foreach_sub(grouped_params, grouped_axs)
        torch._foreach_addcmul_(grouped_axs, intermediate, grouped_mus)
        del intermediate
        if capturable:
            new_mus = torch._foreach_sub(grouped_state_steps, t0)
            torch._foreach_maximum_(new_mus, 1.0)
            torch._foreach_reciprocal_(new_mus)
            torch._foreach_copy_(grouped_mus, new_mus)
            del new_mus
            new_etas = torch._foreach_pow(grouped_state_steps, alpha)
            torch._foreach_mul_(new_etas, lambd)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_add_(new_etas, 1)
            torch._foreach_reciprocal_(new_etas)
            torch._foreach_mul_(new_etas, lr)
            torch._foreach_copy_(grouped_etas, new_etas)
        else:
            step = grouped_state_steps[0].item()
            new_etas = []
            new_mus = []
            for i in range(len(grouped_mus)):
                new_eta = _to_tensor(lr / (1 + lambd * lr * step ** alpha), device=device)
                new_etas.append(new_eta)
                new_mu = _to_tensor(1 / max(1, step - t0), device=device)
                new_mus.append(new_mu)
            torch._foreach_copy_(grouped_etas, new_etas)
            torch._foreach_copy_(grouped_mus, new_mus)