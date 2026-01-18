from typing import Iterable
import cirq
from cirq_google.line.placement.chip import chip_as_adjacency_list, above, below, right_of, left_of
def test_single_qubit():
    q00 = cirq.GridQubit(0, 0)
    assert chip_as_adjacency_list(_create_device([q00])) == {q00: []}