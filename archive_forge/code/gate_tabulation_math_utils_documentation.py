import itertools
from typing import Union, Sequence, Optional
import numpy as np
from cirq.value import random_state
Entanglement fidelity between two unitaries.

    For unitary matrices, this is related to the average unitary fidelity F
    as

    :math:`F = \frac{F_e d + 1}{d + 1}`
    where d is the matrix dimension.

    Args:
        U_actual : Matrix whose fidelity to U_ideal will be computed. This may
            be a non-unitary matrix, i.e. the projection of a larger unitary
            matrix into the computational subspace.
        U_ideal : Unitary matrix to which U_actual will be compared.

    Both arguments may be vectorized, in that their shapes may be of the form
    (...,M,M) (as long as both shapes can be broadcast together).

    Returns:
        The entanglement fidelity between the two unitaries. For inputs with
        shape (...,M,M), the output has shape (...).

    