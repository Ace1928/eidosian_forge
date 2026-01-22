from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
class BackwardTrigger(nn.Module):
    """A backward trigger module.

    This module takes a parameter as an input and create a linked parameter
    from a newly created trigger parameter.

    The way to use it in a module's ``__init__'' and ``forward'' functions:

    ```
    def __init__():
      ...
      self.trigger = BackwardTrigger(some_layer.weight)
      ...

    def forward():
      w = self.trigger()
      ... continue to use w ...
    ```

    As a resule, the trigger's backward hook will be called at the end of
    the backward for the module that uses this trigger.
    """

    def __init__(self, linked_param: torch.Tensor):
        super().__init__()
        assert isinstance(linked_param, nn.Parameter)
        self.trigger = nn.Parameter(torch.rand(1, dtype=linked_param.dtype, device=linked_param.device))
        self.trigger._linked_param = linked_param

    def forward(self) -> torch.Tensor:
        return BackwardTriggerFn.apply(self.trigger._linked_param, self.trigger)