import itertools
import pytest
import sympy
import cirq
@pytest.mark.parametrize('r_list_gen', [lambda: [{'a': 1}, {'a': 1.5}], lambda: [{sympy.Symbol('a'): 1}, {sympy.Symbol('a'): 1.5}], lambda: [cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 1.5})], lambda: [cirq.ParamResolver({sympy.Symbol('a'): 1}), cirq.ParamResolver({sympy.Symbol('a'): 1.5})], lambda: [{'a': 1}, cirq.ParamResolver({sympy.Symbol('a'): 1.5})], lambda: ({'a': 1}, {'a': 1.5}), lambda: (r for r in [{'a': 1}, {'a': 1.5}]), lambda: {object(): r for r in [{'a': 1}, {'a': 1.5}]}.values()])
def test_to_sweep_resolver_list(r_list_gen):
    sweep = cirq.to_sweep(r_list_gen())
    assert isinstance(sweep, cirq.Sweep)
    assert list(sweep) == [cirq.ParamResolver({'a': 1}), cirq.ParamResolver({'a': 1.5})]