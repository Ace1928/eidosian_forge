from sympy.core.backend import symbols, Matrix, atan, zeros
from sympy.simplify.simplify import simplify
from sympy.physics.mechanics import (dynamicsymbols, Particle, Point,
from sympy.testing.pytest import raises
def test_not_specified_errors():
    """This test will cover errors that arise from trying to access attributes
    that were not specified upon object creation or were specified on creation
    and the user tries to recalculate them."""
    symsystem1 = SymbolicSystem(states, comb_explicit_rhs)
    with raises(AttributeError):
        symsystem1.comb_implicit_mat
    with raises(AttributeError):
        symsystem1.comb_implicit_rhs
    with raises(AttributeError):
        symsystem1.dyn_implicit_mat
    with raises(AttributeError):
        symsystem1.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem1.kin_explicit_rhs
    with raises(AttributeError):
        symsystem1.compute_explicit_form()
    symsystem2 = SymbolicSystem(coordinates, comb_implicit_rhs, speeds=speeds, mass_matrix=comb_implicit_mat)
    with raises(AttributeError):
        symsystem2.dyn_implicit_mat
    with raises(AttributeError):
        symsystem2.dyn_implicit_rhs
    with raises(AttributeError):
        symsystem2.kin_explicit_rhs
    with raises(AttributeError):
        symsystem1.coordinates
    with raises(AttributeError):
        symsystem1.speeds
    with raises(AttributeError):
        symsystem1.bodies
    with raises(AttributeError):
        symsystem1.loads
    with raises(AttributeError):
        symsystem2.comb_explicit_rhs