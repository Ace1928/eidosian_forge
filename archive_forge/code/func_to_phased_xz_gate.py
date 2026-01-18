from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
def to_phased_xz_gate(self) -> phased_x_z_gate.PhasedXZGate:
    """Convert this gate to a PhasedXZGate instance.

        The rotation can be categorized by {axis} * {degree}:
            * Identity: I
            * {x, y, z} * {90, 180, 270}  --- {X, Y, Z} + 6 Quarter turn gates
            * {+/-xy, +/-yz, +/-zx} * 180  --- 6 Hadamard-like gates
            * {middle point of xyz in 4 Quadrant} * {120, 240} --- swapping axis
        note 1 + 9 + 6 + 8 = 24 in total.

        To associate with Clifford Tableau, it can also be grouped by 4:
            * {I,X,Y,Z} is [[1 0], [0, 1]]
            * {+/- X_sqrt, 2 Hadamard-like gates acting on the YZ plane} is [[1, 0], [1, 1]]
            * {+/- Z_sqrt, 2 Hadamard-like gates acting on the XY plane} is [[1, 1], [0, 1]]
            * {+/- Y_sqrt, 2 Hadamard-like gates acting on the XZ plane} is [[0, 1], [1, 0]]
            * {middle point of xyz in 4 Quadrant} * 120 is [[0, 1], [1, 1]]
            * {middle point of xyz in 4 Quadrant} * 240 is [[1, 1], [1, 0]]
        """
    x_to_flip, z_to_flip = self.clifford_tableau.rs
    flip_index = int(z_to_flip) * 2 + x_to_flip
    a, x, z = (0.0, 0.0, 0.0)
    matrix = self.clifford_tableau.matrix()
    if np.array_equal(matrix, [[1, 0], [0, 1]]):
        to_phased_xz = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.5, 1.0, 0.0)]
        a, x, z = to_phased_xz[flip_index]
    elif np.array_equal(matrix, [[1, 0], [1, 1]]):
        a = 0.0
        x = 0.5 if x_to_flip ^ z_to_flip else -0.5
        z = 1.0 if x_to_flip else 0.0
    elif np.array_equal(matrix, [[0, 1], [1, 0]]):
        a = 0.5
        x = 0.5 if x_to_flip else -0.5
        z = 0.0 if x_to_flip ^ z_to_flip else 1.0
    elif np.array_equal(matrix, [[1, 1], [0, 1]]):
        to_phased_xz = [(0.0, 0.0, 0.5), (0.0, 0.0, -0.5), (0.25, 1.0, 0.0), (-0.25, 1.0, 0.0)]
        a, x, z = to_phased_xz[flip_index]
    elif np.array_equal(matrix, [[0, 1], [1, 1]]):
        a = 0.5
        x = 0.5 if x_to_flip else -0.5
        z = 0.5 if x_to_flip ^ z_to_flip else -0.5
    else:
        assert np.array_equal(matrix, [[1, 1], [1, 0]])
        a = 0.0
        x = -0.5 if x_to_flip ^ z_to_flip else 0.5
        z = -0.5 if x_to_flip else 0.5
    return phased_x_z_gate.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)