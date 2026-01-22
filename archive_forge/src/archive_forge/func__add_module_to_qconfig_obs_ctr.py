from collections import namedtuple
from typing import Optional, Any, Union, Type
import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
from .observer import (
import warnings
import copy
def _add_module_to_qconfig_obs_ctr(qconfig: QConfigAny, module: Optional[nn.Module]) -> Any:
    """This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig with obs constructors stored in activation and weight
        module: module which the qconfig is related to

    Return:
        qconfig: configured so that obs constructors set to construct on the same device as module
    """
    if module is None or qconfig is None or qconfig._fields != ('activation', 'weight'):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        assert isinstance(module, torch.nn.Module)
        devices = {p.device for p in module.parameters()} | {p.device for p in module.buffers()}
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {'device': device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        try:
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(factory_kwargs=get_factory_kwargs_based_on_module_device)
        except AttributeError:
            return original_constructor
        except TypeError:
            return original_constructor
    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)
    return QConfig(activation, weight)