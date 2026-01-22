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
@dataclass(frozen=True)
class _ZippedCircuit:
    """A fully-wide circuit made by zipping together a bunch of two-qubit circuits
    and its provenance data.

    Args:
        wide_circuit: The zipped circuit on all pairs
        pairs: The pairs of qubits operated on in the wide circuit.
        combination: A list of indices into the (narrow) `circuits` library. Each entry
            indexes the narrow circuit operating on the corresponding pair in `pairs`. This
            is a given row of the combinations matrix. It is essential for being able to
            "unzip" the results of the `wide_circuit`.
        layer_i: Metadata indicating how the `pairs` were generated. This 0-based index is
            which `GridInteractionLayer` or `Moment` was used for these pairs when calibrating
            several spacial layouts in one request. This field does not modify any behavior.
            It is propagated to the output result object.
        combination_i: Metadata indicating how the `wide_circuit` was zipped. This is
            the row index of the combinations matrix that identifies this
            particular combination of component narrow circuits. This field does not modify
            any behavior. It is propagated to the output result object.
    """
    wide_circuit: 'cirq.Circuit'
    pairs: List[Tuple['cirq.Qid', 'cirq.Qid']]
    combination: List[int]
    layer_i: int
    combination_i: int