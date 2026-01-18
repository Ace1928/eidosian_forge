import itertools
import os
import time
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Iterator, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pytest
import sympy
import cirq
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice
def test_pick_inserted_ops_moment_indices():
    for _ in range(20):
        n_moments = randint(1, 10)
        n_qubits = randint(1, 20)
        op_density = random()
        circuit = cirq.testing.random_circuit(n_qubits, n_moments, op_density)
        start = randrange(n_moments)
        first_half = cirq.Circuit(circuit[:start])
        second_half = cirq.Circuit(circuit[start:])
        operations = tuple((op for moment in second_half for op in moment.operations))
        squeezed_second_half = cirq.Circuit(operations, strategy=cirq.InsertStrategy.EARLIEST)
        expected_circuit = cirq.Circuit(first_half._moments + squeezed_second_half._moments)
        expected_circuit._moments += [cirq.Moment() for _ in range(len(circuit) - len(expected_circuit))]
        insert_indices, _ = circuits.circuit._pick_inserted_ops_moment_indices(operations, start)
        actual_circuit = cirq.Circuit(first_half._moments + [cirq.Moment() for _ in range(n_moments - start)])
        for op, insert_index in zip(operations, insert_indices):
            actual_circuit._moments[insert_index] = actual_circuit._moments[insert_index].with_operation(op)
        assert actual_circuit == expected_circuit