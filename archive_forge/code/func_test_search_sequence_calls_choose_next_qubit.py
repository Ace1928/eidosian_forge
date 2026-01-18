from typing import Iterable
from unittest import mock
import pytest
import cirq
from cirq_google.line.placement import greedy
from cirq_google.line.placement.sequence import GridQubitLineTuple, NotFoundError
def test_search_sequence_calls_choose_next_qubit():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q02 = cirq.GridQubit(0, 2)
    qubits = [q00, q01, q02]
    search = greedy.GreedySequenceSearch(_create_device(qubits), q01)
    with mock.patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [])
        choose_next_qubit.assert_called_once_with(q01, {q01})
    with mock.patch.object(search, '_choose_next_qubit') as choose_next_qubit:
        choose_next_qubit.return_value = None
        search._sequence_search(q01, [q00])
        choose_next_qubit.assert_called_once_with(q01, {q00, q01})