import re
import pytest
import sympy
import cirq
def test_sympy_condition_qasm():
    assert cirq.SympyCondition(sympy.Eq(sympy.Symbol('a'), 2)).qasm == 'm_a==2'
    with pytest.raises(ValueError, match='QASM is defined only for SympyConditions of type key == constant'):
        _ = cirq.SympyCondition(sympy.Symbol('a') != 2).qasm