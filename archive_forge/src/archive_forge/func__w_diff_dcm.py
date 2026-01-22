from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _w_diff_dcm(self, otherframe):
    """Angular velocity from time differentiating the DCM. """
    from sympy.physics.vector.functions import dynamicsymbols
    dcm2diff = otherframe.dcm(self)
    diffed = dcm2diff.diff(dynamicsymbols._t)
    angvelmat = diffed * dcm2diff.T
    w1 = trigsimp(expand(angvelmat[7]), recursive=True)
    w2 = trigsimp(expand(angvelmat[2]), recursive=True)
    w3 = trigsimp(expand(angvelmat[3]), recursive=True)
    return Vector([(Matrix([w1, w2, w3]), otherframe)])