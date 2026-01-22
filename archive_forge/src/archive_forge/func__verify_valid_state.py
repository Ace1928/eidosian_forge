from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
def _verify_valid_state(qubits: List[cirq.GridQubit], state: _STATE):
    seqs, edges = state
    search = AnnealSequenceSearch(_create_device(qubits), seed=4027383828)
    c_adj = chip_as_adjacency_list(_create_device(qubits))
    for e in edges:
        assert search._normalize_edge(e) == e
    for n0, n1 in edges:
        assert n0 in c_adj[n1]
    for n0 in c_adj:
        for n1 in c_adj[n0]:
            assert (n0, n1) in edges or (n1, n0) in edges
            assert (n0, n1) in edges or (n1, n0) in edges
    c_set = set(qubits)
    for seq in seqs:
        for n in seq:
            c_set.remove(n)
    assert not c_set