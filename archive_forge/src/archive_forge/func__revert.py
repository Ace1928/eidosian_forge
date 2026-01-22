from typing import TYPE_CHECKING
import torch
from ...utils import logging, recurse_getattr, recurse_setattr
def _revert(self, module: torch.nn.Module) -> torch.nn.Module:
    if self.module_mapping is not None:
        if '' in self.module_mapping.values():
            for bt_module_attr_name, value in self.module_mapping.items():
                if value == '':
                    module = getattr(self, bt_module_attr_name)
                    return module
        else:
            raise NotImplementedError('replacing a submodule in revert is not supported')
    for modified_layer_key_names, original_layer_key_names in self.original_layers_mapping.items():
        if isinstance(original_layer_key_names, list):
            current_weight = getattr(self, modified_layer_key_names)
            split_index = current_weight.shape[0] // len(original_layer_key_names)
            for i, subparam_name in enumerate(original_layer_key_names):
                if recurse_getattr(module, subparam_name) is None:
                    continue
                if module not in self.keys_to_ignore:
                    parameter = current_weight[i * split_index:(i + 1) * split_index].clone()
                    if isinstance(recurse_getattr(module, subparam_name), torch.nn.Parameter):
                        parameter = torch.nn.Parameter(parameter)
                    recurse_setattr(module, subparam_name, parameter)
        elif isinstance(original_layer_key_names, str):
            if recurse_getattr(module, original_layer_key_names) is None:
                continue
            parameter = getattr(self, modified_layer_key_names)
            if isinstance(recurse_getattr(module, original_layer_key_names), torch.nn.Parameter):
                parameter = torch.nn.Parameter(parameter)
            recurse_setattr(module, original_layer_key_names, parameter)
        else:
            raise ValueError(f'Invalid type {type(modified_layer_key_names)} for `original_layers_mapping`', ' please use either `str` or `list`.')
    return module