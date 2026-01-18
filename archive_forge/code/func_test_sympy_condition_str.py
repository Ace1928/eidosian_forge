import re
import pytest
import sympy
import cirq
def test_sympy_condition_str():
    assert str(init_sympy_condition) == '0:a >= 1'