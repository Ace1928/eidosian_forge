import numpy as np
import pennylane as qml
from pennylane import math
def _xyx_decomposition(U, wire, return_global_phase=False):
    """Compute the decomposition of a single-qubit matrix :math:`U` in terms
    of elementary operations, as a product of X and Y rotations in the form
    :math:`e^{i\\gamma} RX(\\phi) RY(\\theta) RX(\\lambda)`.

    Args:
        U (array[complex]): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-gamma)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates, an ``RX``, an ``RY`` and another ``RX`` gate,
            which when applied in the order of appearance in the list is equivalent to the unitary
            :math:`U` up to global phase. If `return_global_phase=True`, the global phase is returned
            as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = _xyx_decomposition(U, 0, return_global_phase=True)
    >>> decompositions
    [RX(10.845351366405708, wires=[0]),
     RY(1.3974974118006183, wires=[0]),
     RX(0.45246583660683803, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]
    """
    EPS = 1e-64
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, gammas = _convert_to_su2(U, return_global_phase=True)
    lams_plus_phis = math.arctan2(-math.imag(U_det1[:, 0, 1]), math.real(U_det1[:, 0, 0]) + EPS)
    lams_minus_phis = math.arctan2(math.imag(U_det1[:, 0, 0]), -math.real(U_det1[:, 0, 1]) + EPS)
    lams = lams_plus_phis + lams_minus_phis
    phis = lams_plus_phis - lams_minus_phis
    thetas = math.where(math.isclose(math.sin(lams_plus_phis), math.zeros_like(lams_plus_phis)), 2 * math.arccos(math.real(U_det1[:, 1, 1]) / (math.cos(lams_plus_phis) + EPS)), 2 * math.arccos(-math.imag(U_det1[:, 0, 1]) / (math.sin(lams_plus_phis) + EPS)))
    phis, thetas, lams, gammas = map(math.squeeze, [phis, thetas, lams, gammas])
    phis = phis % (4 * np.pi)
    thetas = thetas % (4 * np.pi)
    lams = lams % (4 * np.pi)
    operations = [qml.RX(lams, wire), qml.RY(thetas, wire), qml.RX(phis, wire)]
    if return_global_phase:
        operations.append(qml.GlobalPhase(-gammas))
    return operations