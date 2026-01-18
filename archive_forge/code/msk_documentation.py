import mosek
from cvxopt import matrix, spmatrix, sparse
import sys

    Solves the mixed integer LP

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0
                    xi integer, forall i in I

    using MOSEK 8.

    solsta, x = ilp(c, G, h, A=None, b=None, I=None, taskfile=None).

    Input arguments

        G is m x n, h is m x 1, A is p x n, b is p x 1.  G and A must be
        dense or sparse 'd' matrices.   h and b are dense 'd' matrices
        with one column.  The default values for A and b are empty
        matrices with zero rows.

        I is a Python set with indices of integer elements of x.  By
        default all elements in x are constrained to be integer, i.e.,
        the default value of I is I = set(range(n))

        Dual variables are not returned for MOSEK.

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.integer_optimal, then x contains
                the solution.
            If solsta is mosek.solsta.unknown, then x is None.

            Other return values for solsta include:
                mosek.solsta.near_integer_optimal
            in which case the x value may not be well-defined,
            c.f., section 17.48 of the MOSEK Python API manual.

        x is the solution

    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

    >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.
    