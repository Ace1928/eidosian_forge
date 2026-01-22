import concurrent
import os
import time
import uuid
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from typing import (
import numpy as np
import pandas as pd
import tqdm
from cirq import ops, devices, value, protocols
from cirq.circuits import Circuit, Moment
from cirq.experiments.random_quantum_circuit_generation import CircuitLibraryCombination
def _zip_circuits(circuits: Sequence['cirq.Circuit'], combinations_by_layer: List[CircuitLibraryCombination]) -> List[_ZippedCircuit]:
    """Helper function used in `sample_2q_xeb_circuits` to zip together circuits.

    This takes a sequence of narrow `circuits` and "zips" them together according to the
    combinations in `combinations_by_layer`.
    """
    for layer_combinations in combinations_by_layer:
        if np.any(layer_combinations.combinations < 0) or np.any(layer_combinations.combinations >= len(circuits)):
            raise ValueError('`combinations_by_layer` has invalid indices.')
    zipped_circuits: List[_ZippedCircuit] = []
    for layer_i, layer_combinations in enumerate(combinations_by_layer):
        for combination_i, combination in enumerate(layer_combinations.combinations):
            wide_circuit = Circuit.zip(*(circuits[i].transform_qubits(lambda q: pair[q.x]) for i, pair in zip(combination, layer_combinations.pairs)))
            zipped_circuits.append(_ZippedCircuit(wide_circuit=wide_circuit, pairs=layer_combinations.pairs, combination=combination.tolist(), layer_i=layer_i, combination_i=combination_i))
    return zipped_circuits