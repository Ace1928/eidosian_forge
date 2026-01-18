import itertools
import pytest
import sympy
import cirq
def test_to_sweep_type_error():
    with pytest.raises(TypeError, match='Unexpected sweep'):
        cirq.to_sweep(5)