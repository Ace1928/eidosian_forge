import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_sympy_path_prefix():
    q = cirq.LineQubit(0)
    op = cirq.X(q).with_classical_controls(sympy.Symbol('b'))
    prefixed = cirq.with_key_path_prefix(op, ('0',))
    assert cirq.control_keys(prefixed) == {'0:b'}