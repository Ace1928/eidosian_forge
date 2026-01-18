from sympy.core.backend import sympify
from sympy.physics.vector import Point
from sympy.utilities.exceptions import sympy_deprecation_warning
Returns an inertia dyadic of the particle with respect to another
        point and frame.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the particle expressed about the provided
            point and frame.

        