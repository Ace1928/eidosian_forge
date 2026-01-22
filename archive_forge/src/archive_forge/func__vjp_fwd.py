from functools import partial
import jax
import jax.numpy as jnp
import pennylane as qml
from pennylane.typing import ResultBatch
from ..jacobian_products import _compute_jvps
from .jax import _NonPytreeWrapper
def _vjp_fwd(params, tapes, execute_fn, jpc, device):
    """Perform the forward pass execution, return results and the parameters as residuals."""
    return (_execute_wrapper_vjp(params, tapes, execute_fn, jpc, device), params)