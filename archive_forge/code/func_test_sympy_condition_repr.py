import re
import pytest
import sympy
import cirq
def test_sympy_condition_repr():
    cirq.testing.assert_equivalent_repr(init_sympy_condition)