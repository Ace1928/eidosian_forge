import dataclasses
import itertools
from typing import (
import networkx as nx
import numpy as np
from cirq import circuits, devices, ops, protocols, value
from cirq._doc import document
def _get_random_combinations(n_library_circuits: int, n_combinations: int, *, pair_gen: Iterator[Tuple[List[QidPairT], Any]], random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None) -> List[CircuitLibraryCombination]:
    """For qubit pairs, prepare a set of combinations to efficiently sample
    parallel two-qubit XEB circuits.

    This helper function should be called by one of
    `get_random_comibations_for_device`,
    `get_random_combinations_for_layer_circuit`, or
    `get_random_combinations_for_pairs` which define
    appropriate `pair_gen` arguments.

    Args:
        n_library_circuits: The number of circuits in your library. Likely the value
            passed to `generate_library_of_2q_circuits`.
        n_combinations: The number of combinations (with replacement) to generate
            using the library circuits. Since this function returns a
            `CircuitLibraryCombination`, the combinations will be represented
            by indexes between 0 and `n_library_circuits-1` instead of the circuits
            themselves. The more combinations, the more precise of an estimate for XEB
            fidelity estimation, but a corresponding increase in the number of circuits
            you must sample.
        pair_gen: A generator that yields tuples of (pairs, layer_meta) where pairs is a list
            of qubit pairs and layer_meta is additional data describing the "layer" assigned
            to the CircuitLibraryCombination.layer field.
        random_state: A random-state-like object to seed the random combination generation.

    Returns:
        A list of `CircuitLibraryCombination`, each corresponding to a layer
        generated from `pair_gen`. Each object has a `combinations` matrix of circuit
        indices of shape `(n_combinations, len(pairs))`. This
        returned list can be provided to `sample_2q_xeb_circuits` to efficiently
        sample parallel XEB circuits.
    """
    rs = value.parse_random_state(random_state)
    combinations_by_layer = []
    for pairs, layer in pair_gen:
        combinations = rs.randint(0, n_library_circuits, size=(n_combinations, len(pairs)))
        combinations_by_layer.append(CircuitLibraryCombination(layer=layer, combinations=combinations, pairs=pairs))
    return combinations_by_layer