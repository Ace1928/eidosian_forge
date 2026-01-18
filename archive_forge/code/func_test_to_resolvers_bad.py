import itertools
import pytest
import sympy
import cirq
def test_to_resolvers_bad():
    with pytest.raises(TypeError, match='Unrecognized sweepable'):
        for _ in cirq.study.to_resolvers('nope'):
            pass