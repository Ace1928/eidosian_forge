import torch
from torch import Tensor
from .optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach,
from torch._utils import is_compiling
from typing import List, Optional
class ASGD(Optimizer):

    def __init__(self, params, lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0, foreach: Optional[bool]=None, maximize: bool=False, differentiable: bool=False, capturable: bool=False):
        if not 0.0 <= lr:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= weight_decay:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if foreach is False and capturable:
            raise ValueError('Capturable not supported with single tensor ASGD')
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay, foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('foreach', None)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']), dtype=torch.float32)
        eta_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['eta'])
        if not eta_is_tensor:
            for s in state_values:
                s['eta'] = torch.tensor(s['eta'], dtype=torch.float32)
        mu_is_tensor = len(state_values) != 0 and torch.is_tensor(state_values[0]['mu'])
        if not mu_is_tensor:
            for s in state_values:
                s['mu'] = torch.tensor(float(s['mu']), dtype=torch.float32)

    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        has_complex = False
        for p in group['params']:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('ASGD does not support sparse gradients')
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = torch.zeros((), device=p.device, dtype=torch.float32)
                    state['eta'] = torch.tensor(group['lr'], device=p.device, dtype=torch.float32)
                    state['mu'] = torch.ones((), device=p.device, dtype=torch.float32)
                    state['ax'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                mus.append(state['mu'])
                axs.append(state['ax'])
                etas.append(state['eta'])
                state_steps.append(state['step'])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mus = []
            axs = []
            etas = []
            state_steps = []
            has_complex = self._init_group(group, params_with_grad, grads, mus, axs, etas, state_steps)
            asgd(params_with_grad, grads, axs, mus, etas, state_steps, lambd=group['lambd'], lr=group['lr'], t0=group['t0'], alpha=group['alpha'], weight_decay=group['weight_decay'], foreach=group['foreach'], maximize=group['maximize'], differentiable=group['differentiable'], capturable=group['capturable'], has_complex=has_complex)
        return loss