import sympy
import cirq
from cirq.study import flatten_expressions
def test_transform_params():
    qubit = cirq.LineQubit(0)
    a = sympy.Symbol('a')
    circuit = cirq.Circuit(cirq.X(qubit) ** (a / 4), cirq.X(qubit) ** (1 + a / 2))
    params = {'a': 3}
    _, new_params = cirq.flatten_with_params(circuit, params)
    expected_params = {sympy.Symbol('<a/4>'): 3 / 4, sympy.Symbol('<a/2 + 1>'): 1 + 3 / 2}
    assert new_params == expected_params