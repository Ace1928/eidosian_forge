from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.physics.units import Quantity, length, meter
from sympy.physics.units.prefixes import PREFIXES, Prefix, prefix_unit, kilo, \
from sympy.physics.units.systems import SI
def test_prefix_unit():
    m = Quantity('fake_meter', abbrev='m')
    m.set_global_relative_scale_factor(1, meter)
    pref = {'m': PREFIXES['m'], 'c': PREFIXES['c'], 'd': PREFIXES['d']}
    q1 = Quantity('millifake_meter', abbrev='mm')
    q2 = Quantity('centifake_meter', abbrev='cm')
    q3 = Quantity('decifake_meter', abbrev='dm')
    SI.set_quantity_dimension(q1, length)
    SI.set_quantity_scale_factor(q1, PREFIXES['m'])
    SI.set_quantity_scale_factor(q1, PREFIXES['c'])
    SI.set_quantity_scale_factor(q1, PREFIXES['d'])
    res = [q1, q2, q3]
    prefs = prefix_unit(m, pref)
    assert set(prefs) == set(res)
    assert {v.abbrev for v in prefs} == set(symbols('mm,cm,dm'))