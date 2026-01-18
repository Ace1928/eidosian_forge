from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
Returns equations that can be solved numerically.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrices.MatrixBase.inv`

        Returns
        ========

        Matrix
            Numerically solvable equations.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's rhs function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's rhs function.

        