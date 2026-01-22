from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, rot_axis1, rot_axis2, rot_axis3)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector
class ThreeAngleOrienter(Orienter):
    """
    Super-class for Body and Space orienters.
    """

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if isinstance(rot_order, Str):
            rot_order = rot_order.name
        approved_orders = ('123', '231', '312', '132', '213', '321', '121', '131', '212', '232', '313', '323', '')
        original_rot_order = rot_order
        rot_order = str(rot_order).upper()
        if not len(rot_order) == 3:
            raise TypeError('rot_order should be a str of length 3')
        rot_order = [i.replace('X', '1') for i in rot_order]
        rot_order = [i.replace('Y', '2') for i in rot_order]
        rot_order = [i.replace('Z', '3') for i in rot_order]
        rot_order = ''.join(rot_order)
        if rot_order not in approved_orders:
            raise TypeError('Invalid rot_type parameter')
        a1 = int(rot_order[0])
        a2 = int(rot_order[1])
        a3 = int(rot_order[2])
        angle1 = sympify(angle1)
        angle2 = sympify(angle2)
        angle3 = sympify(angle3)
        if cls._in_order:
            parent_orient = _rot(a1, angle1) * _rot(a2, angle2) * _rot(a3, angle3)
        else:
            parent_orient = _rot(a3, angle3) * _rot(a2, angle2) * _rot(a1, angle1)
        parent_orient = parent_orient.T
        obj = super().__new__(cls, angle1, angle2, angle3, Str(rot_order))
        obj._angle1 = angle1
        obj._angle2 = angle2
        obj._angle3 = angle3
        obj._rot_order = original_rot_order
        obj._parent_orient = parent_orient
        return obj

    @property
    def angle1(self):
        return self._angle1

    @property
    def angle2(self):
        return self._angle2

    @property
    def angle3(self):
        return self._angle3

    @property
    def rot_order(self):
        return self._rot_order