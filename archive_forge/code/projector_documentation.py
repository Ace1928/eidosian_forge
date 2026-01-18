import itertools
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix
from cirq import value
from cirq.ops import raw_types
Expectation of the projection from a density matrix.

        Computes the expectation value of this ProjectorString on the provided state.

        Args:
            state: An array representing a valid  density matrix.
            qid_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        