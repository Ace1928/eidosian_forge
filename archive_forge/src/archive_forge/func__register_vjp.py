from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def _register_vjp(state0, state1):
    """
    Register the interface-specific custom VJP based on the interfaces of the given states

    This function is needed because we don't want to register the custom
    VJPs at PennyLane import time.
    """
    interface = qml.math.get_interface(state0, state1)
    if interface == 'jax':
        _register_jax_vjp()
    elif interface == 'torch':
        _register_torch_vjp()
    elif interface == 'tensorflow':
        _register_tf_vjp()