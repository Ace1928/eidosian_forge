from sympy.core.backend import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs
from collections import namedtuple
from collections.abc import Iterable
Linearize the system about the operating point. Note that
        q_op, u_op, qd_op, ud_op must satisfy the equations of motion.
        These may be either symbolic or numeric.

        Parameters
        ==========

        op_point : dict or iterable of dicts, optional
            Dictionary or iterable of dictionaries containing the operating
            point conditions. These will be substituted in to the linearized
            system before the linearization is complete. Leave blank if you
            want a completely symbolic form. Note that any reduction in
            symbols (whether substituted for numbers or expressions with a
            common parameter) will result in faster runtime.

        A_and_B : bool, optional
            If A_and_B=False (default), (M, A, B) is returned for forming
            [M]*[q, u]^T = [A]*[q_ind, u_ind]^T + [B]r. If A_and_B=True,
            (A, B) is returned for forming dx = [A]x + [B]r, where
            x = [q_ind, u_ind]^T.

        simplify : bool, optional
            Determines if returned values are simplified before return.
            For large expressions this may be time consuming. Default is False.

        Potential Issues
        ================

            Note that the process of solving with A_and_B=True is
            computationally intensive if there are many symbolic parameters.
            For this reason, it may be more desirable to use the default
            A_and_B=False, returning M, A, and B. More values may then be
            substituted in to these matrices later on. The state space form can
            then be found as A = P.T*M.LUsolve(A), B = P.T*M.LUsolve(B), where
            P = Linearizer.perm_mat.
        