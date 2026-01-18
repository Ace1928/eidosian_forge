from itertools import combinations, product
from random import randint
from string import ascii_lowercase as alphabet
from typing import Optional, Sequence, Tuple
import numpy
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_network_decomposition():
    qubits = cirq.LineQubit.range(8)
    swap_network_gate = cca.SwapNetworkGate((4, 4), 5)
    operations = cirq.decompose_once_with_qubits(swap_network_gate, qubits)
    circuit = cirq.Circuit(operations)
    expected_text_diagram = '\n0: ───█─────────────█─────────────╲0╱─────────────█─────────█───────0↦2───\n      │             │             │               │         │       │\n1: ───█─────────────█─────────────╲1╱─────────────█─────────█───────1↦3───\n      │             │             │               │         │       │\n2: ───█─────────────█───1↦0───────╲2╱───────1↦0───█─────────█───────2↦0───\n      │             │   │         │         │     │         │       │\n3: ───█───█─────────█───0↦1───█───╲3╱───█───0↦1───█─────────█───█───3↦1───\n      │   │         │         │   │     │         │         │   │   │\n4: ───█───█───0↦1───█─────────█───╱4╲───█─────────█───0↦1───█───█───4↦6───\n          │   │               │   │     │             │         │   │\n5: ───────█───1↦0─────────────█───╱5╲───█─────────────1↦0───────█───5↦7───\n          │                   │   │     │                       │   │\n6: ───────█───────────────────█───╱6╲───█───────────────────────█───6↦4───\n          │                   │   │     │                       │   │\n7: ───────█───────────────────█───╱7╲───█───────────────────────█───7↦5───\n    '.strip()
    ct.assert_has_diagram(circuit, expected_text_diagram)