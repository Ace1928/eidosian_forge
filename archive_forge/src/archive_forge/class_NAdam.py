import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt, _stack_if_compiling,
from typing import List, Optional
class NAdam(Optimizer):

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, decoupled_weight_decay: bool=False, *, foreach: Optional[bool]=None, capturable: bool=False, differentiable: bool=False):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if not 0.0 <= momentum_decay:
            raise ValueError(f'Invalid momentum_decay value: {momentum_decay}')
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, momentum_decay=momentum_decay, decoupled_weight_decay=decoupled_weight_decay, foreach=foreach, capturable=capturable, differentiable=differentiable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('decoupled_weight_decay', False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']), dtype=torch.float32)
        mu_product_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['mu_product'])
        if not mu_product_is_tensor:
            for s in state_values:
                s['mu_product'] = torch.tensor(s['mu_product'], dtype=torch.float32)

    def _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps):
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((), dtype=torch.float32, device=p.device) if group['capturable'] else torch.tensor(0.0, dtype=torch.float32)
                    state['mu_product'] = torch.ones((), dtype=torch.float32, device=p.device) if group['capturable'] else torch.tensor(1.0, dtype=torch.float32)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                mu_products.append(state['mu_product'])
                state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            mu_products = []
            state_steps = []
            beta1, beta2 = group['betas']
            has_complex = self._init_group(group, params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps)
            nadam(params_with_grad, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps, beta1=beta1, beta2=beta2, lr=group['lr'], weight_decay=group['weight_decay'], momentum_decay=group['momentum_decay'], eps=group['eps'], decoupled_weight_decay=group['decoupled_weight_decay'], foreach=group['foreach'], capturable=group['capturable'], differentiable=group['differentiable'], has_complex=has_complex)
        return loss