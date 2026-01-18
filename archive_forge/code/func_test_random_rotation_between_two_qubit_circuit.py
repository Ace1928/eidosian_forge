import itertools
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast
import networkx as nx
import numpy as np
import pytest
import cirq
from cirq.experiments import (
from cirq.experiments.random_quantum_circuit_generation import (
def test_random_rotation_between_two_qubit_circuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = random_rotations_between_two_qubit_circuit(q0, q1, 4, seed=52)
    assert len(circuit) == 4 * 2 + 1
    assert circuit.all_qubits() == {q0, q1}
    circuit = random_rotations_between_two_qubit_circuit(q0, q1, 4, seed=52, add_final_single_qubit_layer=False)
    assert len(circuit) == 4 * 2
    assert circuit.all_qubits() == {q0, q1}
    cirq.testing.assert_has_diagram(circuit, '0             1\n│             │\nY^0.5         X^0.5\n│             │\n@─────────────@\n│             │\nPhX(0.25)^0.5 Y^0.5\n│             │\n@─────────────@\n│             │\nY^0.5         X^0.5\n│             │\n@─────────────@\n│             │\nX^0.5         PhX(0.25)^0.5\n│             │\n@─────────────@\n│             │', transpose=True)