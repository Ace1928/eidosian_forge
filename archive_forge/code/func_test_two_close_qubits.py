from typing import Iterable
import cirq
from cirq_google.line.placement.chip import chip_as_adjacency_list, above, below, right_of, left_of
def test_two_close_qubits():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    assert chip_as_adjacency_list(_create_device([q00, q01])) == {q00: [q01], q01: [q00]}