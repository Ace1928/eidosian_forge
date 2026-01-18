import itertools
import pytest
import sympy
import cirq
def test_to_resolvers_none():
    assert list(cirq.to_resolvers(None)) == [cirq.ParamResolver({})]