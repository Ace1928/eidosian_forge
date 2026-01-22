from typing import Any, Callable, List, Optional, Union
import torch
@torch.no_grad()
def add_grad(self, param: torch.Tensor) -> None:
    """
        Add a new parameter gradient to the bucket. Param.grad becomes a view of this bucket buffer
        """
    assert id(param) not in self._param_ids, 'The same gradients cannot be checked in twice'
    if param.grad is None:
        param.grad = torch.zeros_like(param)
    self._add_grad_as_view(param)
    self._params.append(param)
    self._param_ids.append(id(param))