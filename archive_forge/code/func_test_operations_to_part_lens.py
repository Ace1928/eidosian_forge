from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_operations_to_part_lens():
    qubits = cirq.LineQubit.range(6)
    ops = [cirq.CZ(*qubits[1:3]), cirq.XX(*qubits[3:5])]
    part_lens = cca.gates.operations_to_part_lens(qubits, ops)
    assert part_lens == (1, 2, 2, 1)
    ops = cirq.CZ(qubits[1], qubits[3])
    with pytest.raises(ValueError):
        cca.gates.operations_to_part_lens(qubits, ops)
    ops = [cirq.CZ(*qubits[1:3]), cirq.CZ(*qubits[2:4])]
    with pytest.raises(ValueError):
        cca.gates.operations_to_part_lens(qubits, ops)