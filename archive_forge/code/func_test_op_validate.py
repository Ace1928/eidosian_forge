from typing import AbstractSet, Iterator, Any
import pytest
import numpy as np
import sympy
import cirq
def test_op_validate():
    op = cirq.X(cirq.LineQid(0, 2))
    op2 = cirq.CNOT(*cirq.LineQid.range(2, dimension=2))
    op.validate_args([cirq.LineQid(1, 2)])
    op2.validate_args(cirq.LineQid.range(1, 3, dimension=2))
    with pytest.raises(ValueError, match='Wrong shape'):
        op.validate_args([cirq.LineQid(1, 9)])
    with pytest.raises(ValueError, match='Wrong number'):
        op.validate_args([cirq.LineQid(1, 2), cirq.LineQid(2, 2)])
    with pytest.raises(ValueError, match='Duplicate'):
        op2.validate_args([cirq.LineQid(1, 2), cirq.LineQid(1, 2)])