import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
class PythonFunctionalizeAPI(BaseFunctionalizeAPI):

    def wrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        return torch.utils._pytree.tree_map_only(FunctionalTensor, FunctionalTensor.to_functional, args)

    def unwrap_tensors(self, args: Tuple[Any]) -> Tuple[Any]:
        return torch.utils._pytree.tree_map_only(FunctionalTensor, FunctionalTensor.from_functional, args)

    def functionalize(self, inner_f: Callable) -> Callable:
        return dispatch_functionalize(inner_f)

    def redispatch_to_next(self) -> ContextManager:
        return unset_functional_temporarily()

    def replace(self, input_tensor, output_tensor) -> None:
        assert isinstance(input_tensor, FunctionalTensor)
        assert not isinstance(output_tensor, FunctionalTensor)
        input_tensor.replace_(output_tensor)

    def commit_update(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.commit_update()

    def sync(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.sync()

    def mark_mutation_hidden_from_autograd(self, tensor) -> None:
        assert isinstance(tensor, FunctionalTensor)
        tensor.mark_mutation_hidden_from_autograd()