from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import torch
def _build_fp32_params(self, params: Any) -> None:
    fp32_params = []
    for p in params:
        p32 = torch.nn.Parameter(p.data.float()).to(p.device)
        p32.grad = torch.zeros_like(p32.data)
        fp32_params.append(p32)
    params = fp32_params
    self.fp32_param_groups = []
    param_groups = list(params)
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]
    for param_group in param_groups:
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        params = param_group['params']
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        self.fp32_param_groups.append(param_group)