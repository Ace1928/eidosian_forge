import os
from typing import Union, Optional, Tuple, Any, List, Sized, TypeVar
import itertools
from collections import namedtuple
import parlai.utils.logging as logging
import torch.optim
def _place_modulelist(self, submodule: torch.nn.Module) -> None:
    if not isinstance(submodule, torch.nn.ModuleList):
        return
    if getattr(submodule, 'model_parallel_exempt', False):
        return
    assert isinstance(submodule, torch.nn.ModuleList)
    layers = submodule
    layers.is_model_parallel = True
    keyfunc = self.__device_allocations.__getitem__
    layer_assignments = {k: 0 for k in self.devices}
    for layer_no, layer in enumerate(layers):
        if layer_no == 0:
            mostfree = 'cuda:0'
        else:
            mostfree = min(self.devices, key=keyfunc)
        self.__device_allocations[mostfree] += trainable_parameters(layer) * 32
        layer_assignments[mostfree] += 1
    devices = [d for i, d in enumerate(self.devices[:]) if layer_assignments[d] > 0]
    for layer_no, layer in enumerate(layers):
        layer_gpu = devices[0]
        assert layer_assignments[layer_gpu] > 0
        logging.debug(f'Model Parallel: Assigning {layer_no} to {layer_gpu}')
        layer._mp_gpu = layer_gpu
        layers[layer_no] = layer.to(layer_gpu)
        layer_assignments[layer_gpu] -= 1
        if layer_assignments[layer_gpu] == 0:
            devices.pop(0)