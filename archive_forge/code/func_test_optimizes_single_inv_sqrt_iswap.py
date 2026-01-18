from typing import Optional
import cirq
import pytest
import sympy
import numpy as np
def test_optimizes_single_inv_sqrt_iswap():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SQRT_ISWAP_INV(a, b))
    assert_optimization_not_broken(c)
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.SqrtIswapTargetGateset(), ignore_failures=False)
    assert len([1 for op in c.all_operations() if len(op.qubits) == 2]) == 1