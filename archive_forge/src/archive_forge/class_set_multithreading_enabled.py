from typing import Any, Optional
import torch
from torch.utils._contextlib import (
class set_multithreading_enabled(_DecoratorContextManager):
    """Context-manager that sets multithreaded backwards on or off.

    ``set_multithreading_enabled`` will enable or disable multithreaded backwards based on its argument :attr:`mode`.
    It can be used as a context-manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable multithreaded backwards (``True``), or disable
                     (``False``).

    .. note::
        This API does not apply to :ref:`forward-mode AD <forward-mode-ad>`.

    """

    def __init__(self, mode: bool) -> None:
        self.prev = torch._C._is_multithreading_enabled()
        torch._C._set_multithreading_enabled(mode)
        self.mode = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        torch._C._set_multithreading_enabled(self.prev)

    def clone(self) -> 'set_multithreading_enabled':
        return self.__class__(self.mode)