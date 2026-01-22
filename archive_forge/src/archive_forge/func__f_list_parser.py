from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy.core.backend import (Matrix, sympify, Mul, Derivative, sin, cos,
def _f_list_parser(fl, ref_frame):
    """Parses the provided forcelist composed of items
    of the form (obj, force).
    Returns a tuple containing:
        vel_list: The velocity (ang_vel for Frames, vel for Points) in
                the provided reference frame.
        f_list: The forces.

    Used internally in the KanesMethod and LagrangesMethod classes.

    """

    def flist_iter():
        for pair in fl:
            obj, force = pair
            if isinstance(obj, ReferenceFrame):
                yield (obj.ang_vel_in(ref_frame), force)
            elif isinstance(obj, Point):
                yield (obj.vel(ref_frame), force)
            else:
                raise TypeError('First entry in each forcelist pair must be a point or frame.')
    if not fl:
        vel_list, f_list = ((), ())
    else:
        unzip = lambda l: list(zip(*l)) if l[0] else [(), ()]
        vel_list, f_list = unzip(list(flist_iter()))
    return (vel_list, f_list)