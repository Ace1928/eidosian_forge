import functools
from typing import Dict, List, Mapping, Optional, Union
import torch
import torch.nn as nn
from .state import PartialState
from .utils import (
from .utils.modeling import get_non_persistent_buffers
from .utils.other import recursive_getattr
class AlignDevicesHook(ModelHook):
    """
    A generic `ModelHook` that ensures inputs and model weights are on the same device for the forward pass of the
    associated module, potentially offloading the weights after the forward pass.

    Args:
        execution_device (`torch.device`, *optional*):
            The device on which inputs and model weights should be placed before the forward pass.
        offload (`bool`, *optional*, defaults to `False`):
            Whether or not the weights should be offloaded after the forward pass.
        io_same_device (`bool`, *optional*, defaults to `False`):
            Whether or not the output should be placed on the same device as the input was.
        weights_map (`Mapping[str, torch.Tensor]`, *optional*):
            When the model weights are offloaded, a (potentially lazy) map from param names to the tensor values.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to include the associated module's buffers when offloading.
        place_submodules (`bool`, *optional*, defaults to `False`):
            Whether to place the submodules on `execution_device` during the `init_hook` event.
    """

    def __init__(self, execution_device: Optional[Union[int, str, torch.device]]=None, offload: bool=False, io_same_device: bool=False, weights_map: Optional[Mapping]=None, offload_buffers: bool=False, place_submodules: bool=False, skip_keys: Optional[Union[str, List[str]]]=None, tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]]=None):
        self.execution_device = execution_device
        self.offload = offload
        self.io_same_device = io_same_device
        self.weights_map = weights_map
        self.offload_buffers = offload_buffers
        self.place_submodules = place_submodules
        self.skip_keys = skip_keys
        self.input_device = None
        self.param_original_devices = {}
        self.buffer_original_devices = {}
        self.tied_params_names = set()
        self.tied_params_map = tied_params_map

    def __repr__(self):
        return f'AlignDevicesHook(execution_device={self.execution_device}, offload={self.offload}, io_same_device={self.io_same_device}, offload_buffers={self.offload_buffers}, place_submodules={self.place_submodules}, skip_keys={repr(self.skip_keys)})'

    def init_hook(self, module):
        if self.execution_device == 'meta' or self.execution_device == torch.device('meta'):
            self.tied_params_map = None
        if not self.offload and self.execution_device is not None:
            for name, _ in named_module_tensors(module, recurse=self.place_submodules):
                set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
        elif self.offload:
            self.original_devices = {name: param.device for name, param in named_module_tensors(module, recurse=self.place_submodules)}
            if self.weights_map is None:
                self.weights_map = {name: param.to('cpu') for name, param in named_module_tensors(module, include_buffers=self.offload_buffers, recurse=self.place_submodules)}
            for name, _ in named_module_tensors(module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True):
                if self.tied_params_map is not None and recursive_getattr(module, name).data_ptr() in self.tied_params_map:
                    self.tied_params_names.add(name)
                set_module_tensor_to_device(module, name, 'meta')
            if not self.offload_buffers and self.execution_device is not None:
                for name, _ in module.named_buffers(recurse=self.place_submodules):
                    set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
            elif self.offload_buffers and self.execution_device is not None:
                for name in get_non_persistent_buffers(module, recurse=self.place_submodules):
                    set_module_tensor_to_device(module, name, self.execution_device, tied_params_map=self.tied_params_map)
        return module

    def pre_forward(self, module, *args, **kwargs):
        if self.io_same_device:
            self.input_device = find_device([args, kwargs])
        if self.offload:
            self.tied_pointers_to_remove = set()
            for name, _ in named_module_tensors(module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True):
                fp16_statistics = None
                value = self.weights_map[name]
                if 'weight' in name and name.replace('weight', 'SCB') in self.weights_map.keys():
                    if value.dtype == torch.int8:
                        fp16_statistics = self.weights_map[name.replace('weight', 'SCB')]
                if name in self.tied_params_names and value.data_ptr() not in self.tied_params_map:
                    self.tied_params_map[value.data_ptr()] = {}
                if value is not None and self.tied_params_map is not None and (value.data_ptr() in self.tied_params_map) and (self.execution_device not in self.tied_params_map[value.data_ptr()]):
                    self.tied_pointers_to_remove.add((value.data_ptr(), self.execution_device))
                set_module_tensor_to_device(module, name, self.execution_device, value=value, fp16_statistics=fp16_statistics, tied_params_map=self.tied_params_map)
        return (send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device, skip_keys=self.skip_keys))

    def post_forward(self, module, output):
        if self.offload:
            for name, _ in named_module_tensors(module, include_buffers=self.offload_buffers, recurse=self.place_submodules, remove_non_persistent=True):
                set_module_tensor_to_device(module, name, 'meta')
                if type(module).__name__ == 'Linear8bitLt':
                    module.state.SCB = None
                    module.state.CxB = None
            for value_pointer, device in self.tied_pointers_to_remove:
                del self.tied_params_map[value_pointer][device]
            self.tied_pointers_to_remove = set()
        if self.io_same_device and self.input_device is not None:
            output = send_to_device(output, self.input_device, skip_keys=self.skip_keys)
        return output

    def detach_hook(self, module):
        if self.offload:
            for name, device in self.original_devices.items():
                if device != torch.device('meta'):
                    set_module_tensor_to_device(module, name, device, value=self.weights_map.get(name, None))
        return module