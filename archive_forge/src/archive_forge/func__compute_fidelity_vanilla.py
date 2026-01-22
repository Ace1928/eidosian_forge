from functools import lru_cache
import autograd
import autoray as ar
import pennylane as qml
from .utils import cast
from .quantum import _check_density_matrix, _check_state_vector
def _compute_fidelity_vanilla(density_matrix0, density_matrix1):
    """Compute the fidelity for two density matrices with the same number of wires.

    .. math::
            F( \\rho , \\sigma ) = -\\text{Tr}( \\sqrt{\\sqrt{\\rho} \\sigma \\sqrt{\\rho}})^2
    """
    sqrt_mat = qml.math.sqrt_matrix(density_matrix0)
    sqrt_mat_sqrt = sqrt_mat @ density_matrix1 @ sqrt_mat
    evs = qml.math.eigvalsh(sqrt_mat_sqrt)
    evs = qml.math.real(evs)
    evs = qml.math.where(evs > 0.0, evs, 0)
    trace = qml.math.sum(qml.math.sqrt(evs), -1) ** 2
    return trace