import warnings
from sympy.core.add import Add
from sympy.core.function import (Function, diff)
from sympy.core.numbers import (Number, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate
from sympy.physics.units import (amount_of_substance, area, convert_to, find_unit,
from sympy.physics.units.definitions import (amu, au, centimeter, coulomb,
from sympy.physics.units.definitions.dimension_definitions import (
from sympy.physics.units.prefixes import PREFIXES, kilo
from sympy.physics.units.quantities import PhysicalConstant, Quantity
from sympy.physics.units.systems import SI
from sympy.testing.pytest import raises
def test_quantity_abs():
    v_w1 = Quantity('v_w1')
    v_w2 = Quantity('v_w2')
    v_w3 = Quantity('v_w3')
    v_w1.set_global_relative_scale_factor(1, meter / second)
    v_w2.set_global_relative_scale_factor(1, meter / second)
    v_w3.set_global_relative_scale_factor(1, meter / second)
    expr = v_w3 - Abs(v_w1 - v_w2)
    assert SI.get_dimensional_expr(v_w1) == (length / time).name
    Dq = Dimension(SI.get_dimensional_expr(expr))
    assert SI.get_dimension_system().get_dimensional_dependencies(Dq) == {length: 1, time: -1}
    assert meter == sqrt(meter ** 2)