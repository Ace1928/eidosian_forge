import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
class CppFunctionalizeAPI(BaseFunctionalizeAPI):

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import _wrap_all_tensors_to_functional
        return _wrap_all_tensors_to_functional(args, level=0)

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
        return _unwrap_all_tensors_from_functional(args, reapply_views=_reapply_views())

    def functionalize(self, inner_f: Callable) -> Callable:
        return torch.func.functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        return torch._C._ExcludeDispatchKeyGuard(torch._C.DispatchKeySet(torch._C.DispatchKey.Functionalize))

    def replace(self, input_tensor, output_tensor) -> None:
        torch._functionalize_replace(input_tensor, output_tensor)

    def commit_update(self, tensor) -> None:
        torch._functionalize_commit_update(tensor)

    def sync(self, tensor) -> None:
        torch._functionalize_sync(tensor)

    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        torch._functionalize_mark_mutation_hidden_from_autograd(tensor)