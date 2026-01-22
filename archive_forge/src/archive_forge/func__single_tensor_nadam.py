import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt, _stack_if_compiling,
from typing import List, Optional
def _single_tensor_nadam(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor], exp_avg_sqs: List[Tensor], mu_products: List[Tensor], state_steps: List[Tensor], *, beta1: float, beta2: float, lr: float, weight_decay: float, momentum_decay: float, eps: float, decoupled_weight_decay: bool, capturable: bool, differentiable: bool, has_complex: bool):
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
        if not torch._utils.is_compiling() and capturable:
            assert param.is_cuda and mu_product.is_cuda and step_t.is_cuda or (param.is_xla and mu_product.is_xla and step_t.is_xla), 'If capturable=True, params, mu_products, and state_steps must be CUDA or XLA tensors.'
        step_t += 1
        if capturable:
            step = step_t
        else:
            step = _get_value(step_t)
        bias_correction2 = 1 - beta2 ** step
        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        mu = beta1 * (1.0 - 0.5 * 0.96 ** (step * momentum_decay))
        mu_next = beta1 * (1.0 - 0.5 * 0.96 ** ((step + 1) * momentum_decay))
        mu_product *= mu
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()
        if differentiable or capturable:
            denom = denom.add(eps)
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(grad, denom, value=-lr * (1.0 - mu) / (1.0 - _get_value(mu_product)))
            param.addcdiv_(exp_avg, denom, value=-lr * mu_next / (1.0 - mu_product_next))