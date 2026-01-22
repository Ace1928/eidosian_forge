import inspect
import warnings
from functools import wraps, partial
from typing import Callable, Sequence, Optional, Union, Tuple
import logging
from cachetools import LRUCache, Cache
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.typing import ResultBatch
from .set_shots import set_shots
from .jacobian_products import (
def _get_execution_config(gradient_fn, grad_on_execution, interface, device, device_vjp):
    """Helper function to get the execution config."""
    if gradient_fn is None:
        _gradient_method = None
    elif isinstance(gradient_fn, str):
        _gradient_method = gradient_fn
    else:
        _gradient_method = 'gradient-transform'
    config = qml.devices.ExecutionConfig(interface=interface, gradient_method=_gradient_method, grad_on_execution=None if grad_on_execution == 'best' else grad_on_execution, use_device_jacobian_product=device_vjp)
    if isinstance(device, qml.devices.Device):
        _, config = device.preprocess(config)
    return config