import sys

    Solves a quadratic program

        minimize    (1/2)*x'*P*x + q'*x
        subject to  G*x <= h
                    A*x = b.


    Input arguments.

        P is a n x n dense or sparse 'd' matrix with the lower triangular
        part of P stored in the lower triangle.  Must be positive
        semidefinite.

        q is an n x 1 dense 'd' matrix.

        G is an m x n dense or sparse 'd' matrix.

        h is an m x 1 dense 'd' matrix.

        A is a p x n dense or sparse 'd' matrix.

        b is a p x 1 dense 'd' matrix or None.

        solver is None or 'mosek'.

        The default values for G, h, A and b are empty matrices with
        zero rows.


    Output arguments (default solver).

        Returns a dictionary with keys 'status', 'x', 's', 'y', 'z',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility, 'dual infeasibility', 'primal slack',
        'dual slack'.

        The 'status' field has values 'optimal' or 'unknown'.
        If the status is 'optimal', 'x', 's', 'y', 'z' are an approximate
        solution of the primal and dual optimal solutions

            G*x + s = h,  A*x = b
            P*x + G'*z + A'*y + q = 0
            s >= 0, z >= 0
            s'*z = o.

        If the status is 'unknown', 'x', 's', 'y', 'z' are the last
        iterates before termination.  These satisfy s > 0 and z > 0, but
        are not necessarily feasible.

        The values of the other fields are defined as follows.

        - 'primal objective': the primal objective (1/2)*x'*P*x + q'*x.

        - 'dual objective': the dual objective

              L(x,y,z) = (1/2)*x'*P*x + q'*x + z'*(G*x - h) + y'*(A*x-b).

        - 'gap': the duality gap s'*z.

        - 'relative gap': the relative gap, defined as

              gap / -primal objective

          if the primal objective is negative,

              gap / dual objective

          if the dual objective is positive, and None otherwise.

        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s - h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).


        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || P*x + G'*z + A'*y + q || / max(1, ||q||).

        - 'primal slack': the smallest primal slack, min_k s_k.
        - 'dual slack': the smallest dual slack, min_k z_k.

        If the exit status is 'optimal', then the primal and dual
        infeasibilities are guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The gap is less than
        solvers.options['abstol'] (default 1e-7) or the relative gap is
        less than solvers.options['reltol'] (default 1e-6).

        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.


    Output arguments (MOSEK solver).

        The return dictionary has two additional fields
        'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate', and 'status' field
        can also have the values 'primal infeasible' or 'dual infeasible'.

        If the exit status is 'optimal', the different fields have the
        same meaning as for the default solver, but the the magnitude of
        the residuals and duality gap is controlled by the MOSEK exit
        criteria.  The 'residual as primal infeasibility certificate' and
        'residual as dual infeasibility certificate' are None.

        Status 'primal infeasible'.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              G'*z + A'*y = 0,  h'*z + b'*y = -1,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack min z_k.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.

        Status 'dual infeasible'.
        - 'x', 's' are an approximate proof of dual infeasibility

              P*x = 0,  q'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack min_k s_k .
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || P*x || / max(1, ||q||),
              || G*x + s || / max(1, ||h||),
              || A*x || / max(1, ||b||).


        If status is 'unknown', all the other fields are None.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement']  positive integer (default: 0)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The MOSEK parameters can me modified by adding an entry
        options['mosek'], containing a dictionary with MOSEK
        parameter/value pairs, as described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    