from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_optimizes_tagged_partial_cz():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit((cirq.CZ ** 0.5)(a, b).with_tags('mytag'))
    assert_optimization_not_broken(c)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2, 'It should take 2 CZ gates to decompose a CZ**0.5 gate'