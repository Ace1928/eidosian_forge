from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
def _locate_joint_pos(self, body, joint_pos):
    """Returns the attachment point of a body."""
    if joint_pos is None:
        return body.masscenter
    if not isinstance(joint_pos, (Point, Vector)):
        raise TypeError('Attachment point must be a Point or Vector.')
    if isinstance(joint_pos, Vector):
        point_name = f'{self.name}_{body.name}_joint'
        joint_pos = body.masscenter.locatenew(point_name, joint_pos)
    if not joint_pos.pos_from(body.masscenter).dt(body.frame) == 0:
        raise ValueError('Attachment point must be fixed to the associated body.')
    return joint_pos