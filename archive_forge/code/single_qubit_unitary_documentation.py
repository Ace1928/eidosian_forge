import numpy as np
import pennylane as qml
from pennylane import math
Decompose a one-qubit unitary :math:`U` in terms of elementary operations. (batched operation)

    Any one qubit unitary operation can be implemented up to a global phase by composing RX, RY,
    and RZ gates.

    Currently supported values for ``rotations`` are "rot", "ZYZ", "XYX", "XZX", and "ZXZ".

    Args:
        U (tensor): A :math:`2 \times 2` unitary matrix.
        wire (Union[Wires, Sequence[int] or int]): The wire on which to apply the operation.
        rotations (str): A string defining the sequence of rotations to decompose :math:`U` into.
        return_global_phase (bool): Whether to return the global phase as a ``qml.GlobalPhase(-alpha)``
            as the last element of the returned list of operations.

    Returns:
        list[Operation]: Returns a list of gates which when applied in the order of appearance in
            the list is equivalent to the unitary :math:`U` up to a global phase. If
            ``return_global_phase=True``, the global phase is returned as the last element of the list.

    **Example**

    >>> U = np.array([
    ...     [-0.28829348-0.78829734j, 0.30364367+0.45085995j],
    ...     [ 0.53396245-0.10177564j, 0.76279558-0.35024096j]
    ... ])
    >>> decompositions = one_qubit_decomposition(U, 0, "ZXZ", return_global_phase=True)
    >>> decompositions
    [RZ(10.753478981934784, wires=[0]),
     RX(1.1493817777940707, wires=[0]),
     RZ(3.3038544749132295, wires=[0]),
     GlobalPhase(1.1759220332464762, wires=[])]
    