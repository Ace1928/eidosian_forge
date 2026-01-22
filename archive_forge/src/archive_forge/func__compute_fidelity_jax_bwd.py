from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def _compute_fidelity_jax_bwd(res, grad_out):
    dm0, dm1 = res
    return _compute_fidelity_grad(dm0, dm1, grad_out)