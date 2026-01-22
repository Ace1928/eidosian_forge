import numpy as np
import pennylane as qml
from pennylane import math
def _rot_decomposition(U, wire, return_global_phase=False):
    """Compute the decomposition of a single-qubit matrix :math:`U` in terms of
    elementary operations, as a single :class:`.RZ` gate or a :class:`.Rot` gate.

    Diagonal operations can be converted to a single :class:`.RZ` gate, while non-diagonal
    operations will be converted to a :class:`.Rot` gate that implements the original operation
    up to a global phase in the form :math:`RZ(\\omega) RY(\\theta) RZ(\\phi)`.

    .. warning::

        When used with ``jax.jit``, all unitaries will be converted to :class:`.Rot` gates,
        including those that are diagonal.

    Args:
        U (tensor): A 2 x 2 unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        return_global_phase (bool): Whether to return the global phase ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[qml.Operation]: A ``Rot`` gate on the specified wire that implements ``U``
            up to a global phase, or an equivalent ``RZ`` gate if ``U`` is diagonal. If
            `return_global_phase=True`, the global phase is included as the last element.

    **Example**

    Suppose we would like to apply the following unitary operation:

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])

    For PennyLane devices that cannot natively implement ``QubitUnitary``, we
    can instead recover a ``Rot`` gate that implements the same operation, up
    to a global phase:

    >>> decompositions = _rot_decomposition(U, 0)
    >>> decompositions
    [Rot(12.32427531154459, 1.1493817771511354, 1.733058145303424, wires=[0])]
    """
    U = math.expand_dims(U, axis=0) if len(U.shape) == 2 else U
    U_det1, alphas = _convert_to_su2(U, return_global_phase=True)
    if len(U_det1) == 1 and (not math.is_abstract(U_det1[0])):
        if math.allclose(U_det1[0, 0, 1], 0.0):
            angle = 2 * math.angle(U_det1[0, 1, 1]) % (4 * np.pi)
            operations = [qml.RZ(angle, wires=wire)]
            if return_global_phase:
                operations.append(qml.GlobalPhase(-alphas))
            return operations
    phis, thetas, omegas = _zyz_get_rotation_angles(U_det1)
    operations = [qml.Rot(phis, thetas, omegas, wires=wire)]
    if return_global_phase:
        alphas = math.squeeze(alphas)
        operations.append(qml.GlobalPhase(-alphas))
    return operations