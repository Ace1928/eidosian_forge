from sympy.core.containers import Tuple
from sympy.core.numbers import pi
from sympy.core.power import Pow
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.printing.str import sstr
from sympy.physics.units import (
from sympy.physics.units.util import convert_to, check_dimensions
from sympy.testing.pytest import raises
def test_quantity_simplify_across_dimensions():
    from sympy.physics.units.util import quantity_simplify
    from sympy.physics.units import ampere, ohm, volt, joule, pascal, farad, second, watt, siemens, henry, tesla, weber, hour, newton
    assert quantity_simplify(ampere * ohm, across_dimensions=True, unit_system='SI') == volt
    assert quantity_simplify(6 * ampere * ohm, across_dimensions=True, unit_system='SI') == 6 * volt
    assert quantity_simplify(volt / ampere, across_dimensions=True, unit_system='SI') == ohm
    assert quantity_simplify(volt / ohm, across_dimensions=True, unit_system='SI') == ampere
    assert quantity_simplify(joule / meter ** 3, across_dimensions=True, unit_system='SI') == pascal
    assert quantity_simplify(farad * ohm, across_dimensions=True, unit_system='SI') == second
    assert quantity_simplify(joule / second, across_dimensions=True, unit_system='SI') == watt
    assert quantity_simplify(meter ** 3 / second, across_dimensions=True, unit_system='SI') == meter ** 3 / second
    assert quantity_simplify(joule / second, across_dimensions=True, unit_system='SI') == watt
    assert quantity_simplify(joule / coulomb, across_dimensions=True, unit_system='SI') == volt
    assert quantity_simplify(volt / ampere, across_dimensions=True, unit_system='SI') == ohm
    assert quantity_simplify(ampere / volt, across_dimensions=True, unit_system='SI') == siemens
    assert quantity_simplify(coulomb / volt, across_dimensions=True, unit_system='SI') == farad
    assert quantity_simplify(volt * second / ampere, across_dimensions=True, unit_system='SI') == henry
    assert quantity_simplify(volt * second / meter ** 2, across_dimensions=True, unit_system='SI') == tesla
    assert quantity_simplify(joule / ampere, across_dimensions=True, unit_system='SI') == weber
    assert quantity_simplify(5 * kilometer / hour, across_dimensions=True, unit_system='SI') == 25 * meter / (18 * second)
    assert quantity_simplify(5 * kilogram * meter / second ** 2, across_dimensions=True, unit_system='SI') == 5 * newton