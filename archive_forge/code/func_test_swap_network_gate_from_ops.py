from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_network_gate_from_ops():
    n_qubits = 10
    qubits = cirq.LineQubit.range(n_qubits)
    part_lens = (1, 2, 1, 3, 3)
    operations = [cirq.Z(qubits[0]), cirq.CZ(*qubits[1:3]), cirq.CCZ(*qubits[4:7]), cirq.CCZ(*qubits[7:])]
    acquaintance_size = 3
    swap_network = cca.SwapNetworkGate.from_operations(qubits, operations, acquaintance_size)
    assert swap_network.acquaintance_size == acquaintance_size
    assert swap_network.part_lens == part_lens
    acquaintance_size = 2
    operations = []
    qubits = qubits[:5]
    swap_network = cca.SwapNetworkGate.from_operations(qubits, operations, acquaintance_size, cirq.ZZ)
    circuit = cirq.Circuit(swap_network(*qubits))
    cca.DECOMPOSE_PERMUTATION_GATES(circuit)
    expected_diagram = '\n0: ───█───ZZ────────────█───ZZ────────────█───ZZ───\n      │   │             │   │             │   │\n1: ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───\n               │   │             │   │\n2: ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───\n      │   │             │   │             │   │\n3: ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───█───ZZ───\n               │   │             │   │\n4: ────────────█───ZZ────────────█───ZZ────────────\n'.strip()
    cirq.testing.assert_has_diagram(circuit, expected_diagram)