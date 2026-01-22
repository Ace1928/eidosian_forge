import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
class _set_fwd_grad_enabled(_DecoratorContextManager):

    def __init__(self, mode: bool) -> None:
        self.prev = _is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_fwd_grad_enabled(self.prev)