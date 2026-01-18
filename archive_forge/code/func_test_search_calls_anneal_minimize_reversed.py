from typing import Iterable, List
from unittest import mock
import numpy as np
import pytest
import cirq
from cirq_google.line.placement.anneal import (
from cirq_google.line.placement.chip import chip_as_adjacency_list
@mock.patch('cirq_google.line.placement.optimization.anneal_minimize')
def test_search_calls_anneal_minimize_reversed(anneal_minimize):
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    seqs = [[q01, q00]]
    edges = {(q00, q01)}
    anneal_minimize.return_value = (seqs, edges)
    assert AnnealSequenceSearch(_create_device([]), seed=4027383809).search() == seqs
    anneal_minimize.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY, mock.ANY, trace_func=mock.ANY)