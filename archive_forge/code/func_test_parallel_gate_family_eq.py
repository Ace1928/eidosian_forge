import pytest
import sympy
import cirq
from cirq.ops.gateset_test import CustomX, CustomXPowGate
def test_parallel_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    for name, description in [(None, None), ('Custom Name', 'Custom Description')]:
        eq.add_equality_group(cirq.ParallelGateFamily(CustomX, max_parallel_allowed=2, name=name, description=description), cirq.ParallelGateFamily(cirq.ParallelGate(CustomX, 2), name=name, description=description))
        eq.add_equality_group(cirq.ParallelGateFamily(CustomXPowGate, max_parallel_allowed=2, name=name, description=description))
        eq.add_equality_group(cirq.ParallelGateFamily(CustomX, max_parallel_allowed=5, name=name, description=description), cirq.ParallelGateFamily(cirq.ParallelGate(CustomX, 10), max_parallel_allowed=5, name=name, description=description))