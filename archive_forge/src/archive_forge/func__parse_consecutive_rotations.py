from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def _parse_consecutive_rotations(self, angles, rotation_order):
    """Helper for orient_body_fixed and orient_space_fixed.

        Parameters
        ==========
        angles : 3-tuple of sympifiable
            Three angles in radians used for the successive rotations.
        rotation_order : 3 character string or 3 digit integer
            Order of the rotations. The order can be specified by the strings
            ``'XZX'``, ``'131'``, or the integer ``131``. There are 12 unique
            valid rotation orders.

        Returns
        =======

        amounts : list
            List of sympifiables corresponding to the rotation angles.
        rot_order : list
            List of integers corresponding to the axis of rotation.
        rot_matrices : list
            List of DCM around the given axis with corresponding magnitude.

        """
    amounts = list(angles)
    for i, v in enumerate(amounts):
        if not isinstance(v, Vector):
            amounts[i] = sympify(v)
    approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
    rot_order = translate(str(rotation_order), 'XYZxyz', '123123')
    if rot_order not in approved_orders:
        raise TypeError('The rotation order is not a valid order.')
    rot_order = [int(r) for r in rot_order]
    if not len(amounts) == 3 & len(rot_order) == 3:
        raise TypeError('Body orientation takes 3 values & 3 orders')
    rot_matrices = [self._rot(order, amount) for order, amount in zip(rot_order, amounts)]
    return (amounts, rot_order, rot_matrices)