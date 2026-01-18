import itertools
import pytest
import sympy
import cirq
def test_to_resolvers_single():
    resolver = cirq.ParamResolver({})
    assert list(cirq.to_resolvers(resolver)) == [resolver]
    assert list(cirq.to_resolvers({})) == [resolver]