from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def _get_adjusted_next_layer_bias(next_layer: nn.Module, pruned_biases: Tensor, mask: Tensor) -> nn.Parameter:
    """Returns new adjusted bias for the second supported module"""
    if parametrize.is_parametrized(next_layer):
        parametrization_dict = cast(nn.ModuleDict, next_layer.parametrizations)
        weight_parameterizations = cast(ParametrizationList, parametrization_dict.weight)
        next_weight = weight_parameterizations.original
    else:
        next_weight = cast(Tensor, next_layer.weight)
    scaling_weight = next_weight[:, ~mask]
    if isinstance(next_layer, nn.Conv2d):
        scaling_product = torch.matmul(pruned_biases.reshape(1, -1), torch.transpose(scaling_weight, 1, 2))
        sum_range = list(range(len(scaling_product.shape)))[1:]
        scaled_biases = torch.sum(scaling_product, sum_range)
    elif isinstance(next_layer, nn.Linear):
        scaled_biases = torch.matmul(pruned_biases, torch.transpose(scaling_weight, 0, 1))
    else:
        raise NotImplementedError(f'Type {type(next_layer)} not supported yet.')
    if parametrize.is_parametrized(next_layer) and getattr(next_layer, '_bias', None) is not None:
        adjusted_bias = nn.Parameter(scaled_biases + next_layer._bias)
    elif not parametrize.is_parametrized(next_layer) and next_layer.bias is not None:
        adjusted_bias = nn.Parameter(scaled_biases + next_layer.bias)
    else:
        adjusted_bias = nn.Parameter(scaled_biases)
    return adjusted_bias