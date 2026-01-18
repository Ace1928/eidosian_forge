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
def test_push_frontier_random_circuit():
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = cirq.testing.random_circuit(randint(1, 20), n_moments, random())
        qubits = sorted(circuit.all_qubits())
        early_frontier = {q: randint(0, n_moments) for q in sample(qubits, randint(0, len(qubits)))}
        late_frontier = {q: randint(0, n_moments) for q in sample(qubits, randint(0, len(qubits)))}
        update_qubits = sample(qubits, randint(0, len(qubits)))
        orig_early_frontier = {q: f for q, f in early_frontier.items()}
        orig_moments = [m for m in circuit._moments]
        insert_index, n_new_moments = circuit._push_frontier(early_frontier, late_frontier, update_qubits)
        assert set(early_frontier.keys()) == set(orig_early_frontier.keys())
        for q in set(early_frontier).difference(update_qubits):
            assert early_frontier[q] == orig_early_frontier[q]
        for q, f in late_frontier.items():
            assert orig_early_frontier.get(q, 0) <= late_frontier[q] + n_new_moments
            if f != len(orig_moments):
                assert orig_moments[f] == circuit[f + n_new_moments]
        for q in set(update_qubits).intersection(early_frontier):
            if orig_early_frontier[q] == insert_index:
                assert orig_early_frontier[q] == early_frontier[q]
                assert not n_new_moments or circuit._moments[early_frontier[q]] == cirq.Moment()
            elif orig_early_frontier[q] == len(orig_moments):
                assert early_frontier[q] == len(circuit)
            else:
                assert orig_moments[orig_early_frontier[q]] == circuit._moments[early_frontier[q]]