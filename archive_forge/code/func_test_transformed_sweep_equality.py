import sympy
import cirq
from cirq.study import flatten_expressions
def test_transformed_sweep_equality():
    a = sympy.Symbol('a')
    sweep = cirq.Linspace('a', start=0, stop=3, length=4)
    expr_map = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 4: 'x1'})
    sweep2 = cirq.Linspace(a, start=0, stop=3, length=4)
    expr_map2 = cirq.ExpressionMap({a / 4: 'x0', 1 - a / 4: 'x1'})
    sweep3 = cirq.Linspace(a, start=0, stop=3, length=20)
    expr_map3 = cirq.ExpressionMap({a / 20: 'x0', 1 - a / 20: 'x1'})
    et = cirq.testing.EqualsTester()
    et.make_equality_group(lambda: expr_map.transform_sweep(sweep), lambda: expr_map.transform_sweep(sweep2), lambda: expr_map2.transform_sweep(sweep2))
    et.add_equality_group(expr_map.transform_sweep(sweep3))
    et.add_equality_group(expr_map3.transform_sweep(sweep))
    et.add_equality_group(expr_map3.transform_sweep(sweep3))