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
def _verify_and_get_two_qubits_from_circuits(circuits: Sequence['cirq.Circuit']):
    """Make sure each of the provided circuits uses the same two qubits and return them."""
    all_qubits_set: Set['cirq.Qid'] = set()
    all_qubits_set = all_qubits_set.union(*(circuit.all_qubits() for circuit in circuits))
    all_qubits_list = sorted(all_qubits_set)
    if len(all_qubits_list) != 2:
        raise ValueError('`circuits` should be a sequence of circuits each operating on the same two qubits.')
    return all_qubits_list