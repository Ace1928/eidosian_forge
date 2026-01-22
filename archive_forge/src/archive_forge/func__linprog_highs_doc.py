def _linprog_highs_doc(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, method='highs', callback=None, maxiter=None, disp=False, presolve=True, time_limit=None, dual_feasibility_tolerance=None, primal_feasibility_tolerance=None, ipm_optimality_tolerance=None, simplex_dual_edge_weight_strategy=None, mip_rel_gap=None, **unknown_options):
    """
    Linear programming: minimize a linear objective function subject to linear
    equality and inequality constraints using one of the HiGHS solvers.

    Linear programming solves problems of the following form:

    .. math::

        \\min_x \\ & c^T x \\\\
        \\mbox{such that} \\ & A_{ub} x \\leq b_{ub},\\\\
        & A_{eq} x = b_{eq},\\\\
        & l \\leq x \\leq u ,

    where :math:`x` is a vector of decision variables; :math:`c`,
    :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    Alternatively, that's:

    minimize::

        c @ x

    such that::

        A_ub @ x <= b_ub
        A_eq @ x == b_eq
        lb <= x <= ub

    Note that by default ``lb = 0`` and ``ub = None`` unless specified with
    ``bounds``.

    Parameters
    ----------
    c : 1-D array
        The coefficients of the linear objective function to be minimized.
    A_ub : 2-D array, optional
        The inequality constraint matrix. Each row of ``A_ub`` specifies the
        coefficients of a linear inequality constraint on ``x``.
    b_ub : 1-D array, optional
        The inequality constraint vector. Each element represents an
        upper bound on the corresponding value of ``A_ub @ x``.
    A_eq : 2-D array, optional
        The equality constraint matrix. Each row of ``A_eq`` specifies the
        coefficients of a linear equality constraint on ``x``.
    b_eq : 1-D array, optional
        The equality constraint vector. Each element of ``A_eq @ x`` must equal
        the corresponding element of ``b_eq``.
    bounds : sequence, optional
        A sequence of ``(min, max)`` pairs for each element in ``x``, defining
        the minimum and maximum values of that decision variable. Use ``None``
        to indicate that there is no bound. By default, bounds are
        ``(0, None)`` (all decision variables are non-negative).
        If a single tuple ``(min, max)`` is provided, then ``min`` and
        ``max`` will serve as bounds for all decision variables.
    method : str

        This is the method-specific documentation for 'highs', which chooses
        automatically between
        :ref:`'highs-ds' <optimize.linprog-highs-ds>` and
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
        :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
        :ref:`'simplex' <optimize.linprog-simplex>` (legacy)
        are also available.
    integrality : 1-D array or int, optional
        Indicates the type of integrality constraint on each decision variable.

        ``0`` : Continuous variable; no integrality constraint.

        ``1`` : Integer variable; decision variable must be an integer
        within `bounds`.

        ``2`` : Semi-continuous variable; decision variable must be within
        `bounds` or take value ``0``.

        ``3`` : Semi-integer variable; decision variable must be an integer
        within `bounds` or take value ``0``.

        By default, all variables are continuous.

        For mixed integrality constraints, supply an array of shape `c.shape`.
        To infer a constraint on each decision variable from shorter inputs,
        the argument will be broadcasted to `c.shape` using `np.broadcast_to`.

        This argument is currently used only by the ``'highs'`` method and
        ignored otherwise.

    Options
    -------
    maxiter : int
        The maximum number of iterations to perform in either phase.
        For :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`, this does not
        include the number of crossover iterations. Default is the largest
        possible value for an ``int`` on the platform.
    disp : bool (default: ``False``)
        Set to ``True`` if indicators of optimization status are to be
        printed to the console during optimization.
    presolve : bool (default: ``True``)
        Presolve attempts to identify trivial infeasibilities,
        identify trivial unboundedness, and simplify the problem before
        sending it to the main solver. It is generally recommended
        to keep the default setting ``True``; set to ``False`` if
        presolve is to be disabled.
    time_limit : float
        The maximum time in seconds allotted to solve the problem;
        default is the largest possible value for a ``double`` on the
        platform.
    dual_feasibility_tolerance : double (default: 1e-07)
        Dual feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``primal_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    primal_feasibility_tolerance : double (default: 1e-07)
        Primal feasibility tolerance for
        :ref:`'highs-ds' <optimize.linprog-highs-ds>`.
        The minimum of this and ``dual_feasibility_tolerance``
        is used for the feasibility tolerance of
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
    ipm_optimality_tolerance : double (default: ``1e-08``)
        Optimality tolerance for
        :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`.
        Minimum allowable value is 1e-12.
    simplex_dual_edge_weight_strategy : str (default: None)
        Strategy for simplex dual edge weights. The default, ``None``,
        automatically selects one of the following.

        ``'dantzig'`` uses Dantzig's original strategy of choosing the most
        negative reduced cost.

        ``'devex'`` uses the strategy described in [15]_.

        ``steepest`` uses the exact steepest edge strategy as described in
        [16]_.

        ``'steepest-devex'`` begins with the exact steepest edge strategy
        until the computation is too costly or inexact and then switches to
        the devex method.

        Currently, ``None`` always selects ``'steepest-devex'``, but this
        may change as new options become available.
    mip_rel_gap : double (default: None)
        Termination criterion for MIP solver: solver will terminate when the
        gap between the primal objective value and the dual objective bound,
        scaled by the primal objective value, is <= mip_rel_gap.
    unknown_options : dict
        Optional arguments not used by this particular solver. If
        ``unknown_options`` is non-empty, a warning is issued listing
        all unused options.

    Returns
    -------
    res : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields:

        x : 1D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1D array
            The (nominally positive) values of the slack,
            ``b_ub - A_ub @ x``.
        con : 1D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration or time limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : The HiGHS solver ran into a problem.

        message : str
            A string descriptor of the exit status of the algorithm.
        nit : int
            The total number of iterations performed.
            For the HiGHS simplex method, this includes iterations in all
            phases. For the HiGHS interior-point method, this does not include
            crossover iterations.
        crossover_nit : int
            The number of primal/dual pushes performed during the
            crossover routine for the HiGHS interior-point method.
            This is ``0`` for the HiGHS simplex method.
        ineqlin : OptimizeResult
            Solution and sensitivity information corresponding to the
            inequality constraints, `b_ub`. A dictionary consisting of the
            fields:

            residual : np.ndnarray
                The (nominally positive) values of the slack variables,
                ``b_ub - A_ub @ x``.  This quantity is also commonly
                referred to as "slack".

            marginals : np.ndarray
                The sensitivity (partial derivative) of the objective
                function with respect to the right-hand side of the
                inequality constraints, `b_ub`.

        eqlin : OptimizeResult
            Solution and sensitivity information corresponding to the
            equality constraints, `b_eq`.  A dictionary consisting of the
            fields:

            residual : np.ndarray
                The (nominally zero) residuals of the equality constraints,
                ``b_eq - A_eq @ x``.

            marginals : np.ndarray
                The sensitivity (partial derivative) of the objective
                function with respect to the right-hand side of the
                equality constraints, `b_eq`.

        lower, upper : OptimizeResult
            Solution and sensitivity information corresponding to the
            lower and upper bounds on decision variables, `bounds`.

            residual : np.ndarray
                The (nominally positive) values of the quantity
                ``x - lb`` (lower) or ``ub - x`` (upper).

            marginals : np.ndarray
                The sensitivity (partial derivative) of the objective
                function with respect to the lower and upper
                `bounds`.

    Notes
    -----

    Method :ref:`'highs-ds' <optimize.linprog-highs-ds>` is a wrapper
    of the C++ high performance dual revised simplex implementation (HSOL)
    [13]_, [14]_. Method :ref:`'highs-ipm' <optimize.linprog-highs-ipm>`
    is a wrapper of a C++ implementation of an **i**\\ nterior-\\ **p**\\ oint
    **m**\\ ethod [13]_; it features a crossover routine, so it is as accurate
    as a simplex solver. Method :ref:`'highs' <optimize.linprog-highs>` chooses
    between the two automatically. For new code involving `linprog`, we
    recommend explicitly choosing one of these three method values instead of
    :ref:`'interior-point' <optimize.linprog-interior-point>` (default),
    :ref:`'revised simplex' <optimize.linprog-revised_simplex>`, and
    :ref:`'simplex' <optimize.linprog-simplex>` (legacy).

    The result fields `ineqlin`, `eqlin`, `lower`, and `upper` all contain
    `marginals`, or partial derivatives of the objective function with respect
    to the right-hand side of each constraint. These partial derivatives are
    also referred to as "Lagrange multipliers", "dual values", and
    "shadow prices". The sign convention of `marginals` is opposite that
    of Lagrange multipliers produced by many nonlinear solvers.

    References
    ----------
    .. [13] Huangfu, Q., Galabova, I., Feldmeier, M., and Hall, J. A. J.
           "HiGHS - high performance software for linear optimization."
           https://highs.dev/
    .. [14] Huangfu, Q. and Hall, J. A. J. "Parallelizing the dual revised
           simplex method." Mathematical Programming Computation, 10 (1),
           119-142, 2018. DOI: 10.1007/s12532-017-0130-5
    .. [15] Harris, Paula MJ. "Pivot selection methods of the Devex LP code."
            Mathematical programming 5.1 (1973): 1-28.
    .. [16] Goldfarb, Donald, and John Ker Reid. "A practicable steepest-edge
            simplex algorithm." Mathematical Programming 12.1 (1977): 361-371.
    """
    pass