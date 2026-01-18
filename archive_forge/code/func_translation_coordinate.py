from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
@property
def translation_coordinate(self):
    """Generalized coordinate corresponding to the translation distance."""
    return self.coordinates[1]