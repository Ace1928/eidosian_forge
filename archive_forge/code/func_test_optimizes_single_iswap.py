from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_optimizes_single_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.ISWAP(a, b))
    assert_optimization_not_broken(c)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 2