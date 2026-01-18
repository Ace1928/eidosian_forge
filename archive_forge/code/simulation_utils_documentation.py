from typing import Sequence, Tuple
import numpy as np
from cirq import linalg
Returns the probabilities for a state/measurement on the given indices.

    Args:
        state_probability: The multi-qubit state probability vector. This is an
            array of 2 to the power of the number of real numbers, and
            so state must be of size ``2**integer``.  The `state_probability` can be
            a vector of size ``2**integer`` or a tensor of shape
            ``(2, 2, ..., 2)``.
        indices: Which qubits are measured. The `state_probability` is assumed to be
            supplied in big endian order. That is the xth index of v, when
            expressed as a bitstring, has its largest values in the 0th index.
        qid_shape: The qid shape of the `state_probability`.

    Returns:
        State probabilities.
    