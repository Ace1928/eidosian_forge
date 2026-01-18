from typing import Iterable
import cirq
from cirq_google.line.placement.chip import chip_as_adjacency_list, above, below, right_of, left_of
def test_two_qubits_apart():
    q00 = cirq.GridQubit(0, 0)
    q11 = cirq.GridQubit(1, 1)
    assert chip_as_adjacency_list(_create_device([q00, q11])) == {q00: [], q11: []}