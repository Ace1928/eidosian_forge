from typing import Any, Callable, List, Optional, Union
import torch
@torch.no_grad()
def _add_grad_as_view(self, param: torch.Tensor, keep_existing_value: bool=True) -> None:
    assert self.buffer.numel() > 0, 'Cannot add a gradient to a collapsed bucket, please rebuild'
    assert param.dtype == self.buffer.dtype
    assert param.device == self.buffer.device
    fill_next = self._fill + param.numel()
    assert fill_next <= self.buffer.numel()
    if param.grad is not None:
        if keep_existing_value:
            self.buffer[self._fill:fill_next].copy_(param.grad.data.flatten())
        param.grad.data = self.buffer[self._fill:fill_next].view_as(param.data)
    else:
        param.grad = self.buffer[self._fill:fill_next].view_as(param.data)
    self._fill = fill_next