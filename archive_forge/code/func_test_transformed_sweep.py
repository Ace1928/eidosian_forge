import sympy
import cirq
from cirq.study import flatten_expressions
def test_transformed_sweep():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    expr_map = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 2: 'x1'})
    transformed = expr_map.transform_sweep(sweep)
    assert len(transformed) == 4
    assert transformed.keys == ['x0', 'x1']
    params = list(transformed.param_tuples())
    assert len(params) == 4
    assert params[1] == (('x0', 1 / 4), ('x1', 1 - 1 / 2))