import sympy
import cirq
from cirq.study import flatten_expressions
def test_transform_sweep():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.X(qubit) ** (a / 4), cirq.X(qubit) ** (1 + a / 2))
    sweep = cirq.Linspace(a, start=0, stop=3, length=4)
    _, new_sweep = cirq.flatten_with_sweep(circuit, sweep)
    assert isinstance(new_sweep, cirq.Sweep)
    resolvers = list(new_sweep)
    expected_resolvers = [cirq.ParamResolver({'<a/4>': 0.0, '<a/2 + 1>': 1.0}), cirq.ParamResolver({'<a/4>': 0.25, '<a/2 + 1>': 1.5}), cirq.ParamResolver({'<a/4>': 0.5, '<a/2 + 1>': 2}), cirq.ParamResolver({'<a/4>': 0.75, '<a/2 + 1>': 2.5})]
    assert resolvers == expected_resolvers