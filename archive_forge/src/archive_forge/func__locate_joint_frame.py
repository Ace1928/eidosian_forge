from abc import ABC, abstractmethod
from sympy.core.backend import pi, AppliedUndef, Derivative, Matrix
from sympy.physics.mechanics.body import Body
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning
def _locate_joint_frame(self, body, interframe):
    """Returns the attachment frame of a body."""
    if interframe is None:
        return body.frame
    if isinstance(interframe, Vector):
        interframe = Joint._create_aligned_interframe(body, interframe, frame_name=f'{self.name}_{body.name}_int_frame')
    elif not isinstance(interframe, ReferenceFrame):
        raise TypeError('Interframe must be a ReferenceFrame.')
    if not interframe.ang_vel_in(body.frame) == 0:
        raise ValueError(f'Interframe {interframe} is not fixed to body {body}.')
    body.masscenter.set_vel(interframe, 0)
    return interframe