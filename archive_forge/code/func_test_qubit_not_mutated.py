from typing import Iterable
import cirq
from cirq_google.line.placement.chip import chip_as_adjacency_list, above, below, right_of, left_of
def test_qubit_not_mutated():
    qubit = cirq.GridQubit(0, 0)
    above(qubit)
    assert qubit == cirq.GridQubit(0, 0)
    below(qubit)
    assert qubit == cirq.GridQubit(0, 0)
    right_of(qubit)
    assert qubit == cirq.GridQubit(0, 0)
    left_of(qubit)
    assert qubit == cirq.GridQubit(0, 0)