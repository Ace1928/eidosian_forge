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
def _get_ml_boundary_execute(interface: str, grad_on_execution: bool, device_vjp: bool=False, differentiable=False) -> Callable:
    """Imports and returns the function that binds derivatives of the required ml framework.

    Args:
        interface (str): The designated ml framework.

        grad_on_execution (bool): whether or not the device derivatives are taken upon execution
    Returns:
        Callable

    Raises:
        pennylane.QuantumFunctionError if the required package is not installed.

    """
    mapped_interface = INTERFACE_MAP[interface]
    try:
        if mapped_interface == 'autograd':
            from .interfaces.autograd import autograd_execute as ml_boundary
        elif mapped_interface == 'tf':
            if 'autograph' in interface:
                from .interfaces.tensorflow_autograph import execute as ml_boundary
                ml_boundary = partial(ml_boundary, grad_on_execution=grad_on_execution)
            else:
                from .interfaces.tensorflow import tf_execute as full_ml_boundary
                ml_boundary = partial(full_ml_boundary, differentiable=differentiable)
        elif mapped_interface == 'torch':
            from .interfaces.torch import execute as ml_boundary
        elif interface == 'jax-jit':
            if device_vjp:
                from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
            else:
                from .interfaces.jax_jit import jax_jit_jvp_execute as ml_boundary
        elif device_vjp:
            from .interfaces.jax_jit import jax_jit_vjp_execute as ml_boundary
        else:
            from .interfaces.jax import jax_jvp_execute as ml_boundary
    except ImportError as e:
        raise qml.QuantumFunctionError(f"{mapped_interface} not found. Please install the latest version of {mapped_interface} to enable the '{mapped_interface}' interface.") from e
    return ml_boundary