from typing import Iterable, List, Sequence, Tuple, Optional, cast, TYPE_CHECKING
import numpy as np
from cirq.linalg import predicates
from cirq.linalg.decompositions import num_cnots_required, extract_right_diag
from cirq import ops, linalg, protocols, circuits
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phased_x_and_z
from cirq.transformers.eject_z import eject_z
from cirq.transformers.eject_phased_paulis import eject_phased_paulis
def _is_trivial_angle(rad: float, atol: float) -> bool:
    """Tests if a circuit for an operator exp(i*rad*XX) (or YY, or ZZ) can
    be performed with a whole CZ.

    Args:
        rad: The angle in radians, assumed to be in the range [-pi/4, pi/4]
        atol: The absolute tolerance with which to make this judgement.
    """
    return abs(rad) < atol or abs(abs(rad) - np.pi / 4) < atol