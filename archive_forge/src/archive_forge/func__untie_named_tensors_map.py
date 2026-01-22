import contextlib
import warnings
from collections import defaultdict
from typing import Any, Dict, Iterator, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor
def _untie_named_tensors_map(module: 'torch.nn.Module', parameters_and_buffers: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Unties all tied tensors in the module to parameters_and_buffers.

    This function returns a new untied_parameters_and_buffers dictionary and leave the original
    untied_parameters_and_buffers dictionary unchanged. It adds new (missing) keys for tied tensors
    in the module to untied_parameters_and_buffers. The value of the new key is the user-given value
    in the original parameters_and_buffers dictionary.

    If there are more than one user-given values for the same tied tensor, it will raise an error.

    For example, if the module has two tied weights self.foo and self.tied_foo and the user passes
    {'foo': foo_value, ...}, this will return {'foo': foo_value, 'tied_foo': foo_value, ...}. If the
    user passes {'foo': foo_value, 'tied_foo': tied_foo_value, ...}, it will raise an error. If the
    user passes {'foo': foo_value, 'tied_foo': foo_value, ...}, it will not raise an error.

    Args:
        module (torch.nn.Module): the module to determine which tensors are tied.
        parameters_and_buffers (Dict[str, Tensor]): a map of {name: tensor} for reparamaterizing the module.

    Returns:
        A new untied version of the parameters_and_buffers dictionary.

    Raises:
        ValueError: if there are more than one user-given values for the same tied tensor.
    """
    all_named_tensors: Dict[str, Tensor] = {}
    all_named_tensors.update(module.named_parameters(remove_duplicate=False))
    all_named_tensors.update(module.named_buffers(remove_duplicate=False))
    tensor_to_tied_names_map: Dict[Tensor, Set[str]] = defaultdict(set)
    for name, tensor in all_named_tensors.items():
        tensor_to_tied_names_map[tensor].add(name)
    tied_names_map: Dict[str, Set[str]] = {}
    for tied_names in tensor_to_tied_names_map.values():
        if len(tied_names) > 1:
            for tied_name in tied_names:
                tied_names_map[tied_name] = tied_names
    given_names = set(parameters_and_buffers.keys())
    given_names_for_tied_tensors = given_names.intersection(tied_names_map.keys())
    for given_name in given_names_for_tied_tensors:
        tied_names = tied_names_map[given_name]
        if len(tied_names.intersection(given_names_for_tied_tensors)) > 1 and len({parameters_and_buffers[tied_name] for tied_name in tied_names}) != 1:
            raise ValueError(f'functional_call got multiple values for keys {sorted(tied_names)}, which are tied. Consider using tie_weights=False')
    untied_parameters_and_buffers = parameters_and_buffers.copy()
    for given_name in given_names_for_tied_tensors:
        for tied_name in tied_names_map[given_name]:
            untied_parameters_and_buffers[tied_name] = parameters_and_buffers[given_name]
    return untied_parameters_and_buffers