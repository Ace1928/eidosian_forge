import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode

    Functional tensors represent tensors that will remove mutations
    from a program. If you perform a mutable operation on a functional tensor,
    it will re-dispatch to the functional variant of that operation.

    Historically, functionalization is implemented in C++ in the dispatcher.
    This class is a lightweight python shim around the C++ functionalization logic.

    FunctionalTensor is required to be used with a corresponding
    FunctionalTensormode active, because it relies
    on using the mode for dispatch (which can properly handle factory functions).
    