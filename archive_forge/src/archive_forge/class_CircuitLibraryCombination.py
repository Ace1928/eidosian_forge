import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
@dataclasses.dataclass(frozen=True)
class CircuitLibraryCombination:
    """For a given layer (specifically, a set of pairs of qubits), `combinations` is a 2d array
    of shape (n_combinations, len(pairs)) where each row represents a combination (with replacement)
    of two-qubit circuits. The actual values are indices into a list of library circuits.

    `layer` is used for record-keeping. This is the GridInteractionLayer if using
    `get_random_combinations_for_device`, the Moment if using
    `get_random_combinations_for_layer_circuit` and ommitted if using
    `get_random_combinations_for_pairs`.
    """
    layer: Optional[Any]
    combinations: np.ndarray
    pairs: List[QidPairT]