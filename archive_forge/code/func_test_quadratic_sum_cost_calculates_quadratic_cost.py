from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def test_quadratic_sum_cost_calculates_quadratic_cost():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    q03 = cirq.GridQubit(0, 3)

    def calculate_cost(seqs: List[List[cirq.GridQubit]]):
        qubits: List[cirq.GridQubit] = []
        for seq in seqs:
            qubits += seq
        return AnnealSequenceSearch(_create_device(qubits), seed=4027383811)._quadratic_sum_cost((seqs, set()))
    assert np.isclose(calculate_cost([[q00]]), -1.0)
    assert np.isclose(calculate_cost([[q00, q01]]), -1.0)
    assert np.isclose(calculate_cost([[q00], [q01]]), -(0.5 ** 2 + 0.5 ** 2))
    assert np.isclose(calculate_cost([[q00], [q01, q02, q03]]), -(0.25 ** 2 + 0.75 ** 2))