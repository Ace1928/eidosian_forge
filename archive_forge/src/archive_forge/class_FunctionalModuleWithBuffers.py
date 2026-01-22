import copy
from typing import (
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """

    def __init__(self, stateless_model: nn.Module, param_names: Tuple[str, ...], buffer_names: Tuple[str, ...], param_names_map: Dict[str, List[str]], buffer_names_map: Dict[str, List[str]]) -> None:
        super().__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def _create_from(model: nn.Module, disable_autograd_tracking: bool=False) -> Tuple['FunctionalModuleWithBuffers', Tuple[Tensor, ...], Tuple[Tensor, ...]]:
        model_copy = copy.deepcopy(model)
        params, param_names, param_names_map = extract_weights(model_copy)
        buffers, buffer_names, buffer_names_map = extract_buffers(model_copy)
        if disable_autograd_tracking:
            for param in params:
                param.requires_grad_(False)
        return (FunctionalModuleWithBuffers(model_copy, param_names, buffer_names, param_names_map, buffer_names_map), params, buffers)

    def forward(self, params: Iterable[Tensor], buffers: Iterable[Tensor], *args, **kwargs) -> Any:
        old_state = _swap_state(self.stateless_model, self.all_names_map, tuple(params) + tuple(buffers))
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            _swap_state(self.stateless_model, self.all_names_map, old_state)