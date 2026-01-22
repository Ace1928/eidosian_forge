import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
class BaseFunctionalizeAPI(ABC):

    @abstractmethod
    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        pass

    @abstractmethod
    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        pass

    @abstractmethod
    def functionalize(self, inner_f: Callable) -> Callable:
        pass

    @abstractmethod
    def redispatch_to_next(self) -> ContextManager:
        pass

    @abstractmethod
    def replace(self, input_tensor, output_tensor) -> None:
        pass

    @abstractmethod
    def commit_update(self, tensor) -> None:
        pass

    @abstractmethod
    def sync(self, tensor) -> None:
        pass

    @abstractmethod
    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        pass