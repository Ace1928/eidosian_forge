from typing import Any, Optional
import torch
from torch.utils._contextlib import (
class inference_mode(_DecoratorContextManager):
    """Context-manager that enables or disables inference mode.

    InferenceMode is a new context manager analogous to :class:`~no_grad`
    to be used when you are certain your operations will have no interactions
    with autograd (e.g., model training). Code run under this mode gets better
    performance by disabling view tracking and version counter bumps. Note that
    unlike some other mechanisms that locally enable or disable grad,
    entering inference_mode also disables to :ref:`forward-mode AD <forward-mode-ad>`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    .. note::
        Inference mode is one of several mechanisms that can enable or
        disable gradients locally see :ref:`locally-disable-grad-doc` for
        more information on how they compare.

    Args:
        mode (bool or function): Either a boolean flag whether to enable or
            disable inference mode or a Python function to decorate with
            inference mode enabled

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> import torch
        >>> x = torch.ones(1, 2, 3, requires_grad=True)
        >>> with torch.inference_mode():
        ...     y = x * x
        >>> y.requires_grad
        False
        >>> # xdoctest: +SKIP("want string isnt quite right")
        >>> y._version
        Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
        RuntimeError: Inference tensors do not track version counter.
        >>> @torch.inference_mode()
        ... def func(x):
        ...     return x * x
        >>> out = func(x)
        >>> out.requires_grad
        False
        >>> @torch.inference_mode
        ... def doubler(x):
        ...     return x * 2
        >>> out = doubler(x)
        >>> out.requires_grad
        False

    """

    def __init__(self, mode: bool=True) -> None:
        if not torch._jit_internal.is_scripting():
            super().__init__()
        self._inference_mode_raii_context: Optional[torch._C._InferenceMode] = None
        self.mode = mode

    def __new__(cls, mode=True):
        if isinstance(mode, bool):
            return super().__new__(cls)
        return cls()(mode)

    def __enter__(self) -> None:
        self._inference_mode_context = torch._C._InferenceMode(self.mode)
        self._inference_mode_context.__enter__()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._inference_mode_context.__exit__(exc_type, exc_value, traceback)

    def clone(self) -> 'inference_mode':
        return self.__class__(self.mode)