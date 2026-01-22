class SeparationLoopResults:
    """
    Container for results of all separation problems solved
    to a single desired optimality target (local or global).

    Parameters
    ----------
    solved_globally : bool
        True if separation problems were solved to global optimality,
        False otherwise.
    solver_call_results : ComponentMap
        Mapping from performance constraints to corresponding
        ``SeparationSolveCallResults`` objects.
    worst_case_perf_con : None or int, optional
        Performance constraint mapped to ``SeparationSolveCallResults``
        object in `self` corresponding to maximally violating
        separation problem solution.

    Attributes
    ----------
    solver_call_results
    solved_globally
    worst_case_perf_con
    found_violation
    violating_param_realization
    scaled_violations
    violating_separation_variable_values
    subsolver_error
    time_out
    """

    def __init__(self, solved_globally, solver_call_results, worst_case_perf_con):
        """Initialize self (see class docstring)."""
        self.solver_call_results = solver_call_results
        self.solved_globally = solved_globally
        self.worst_case_perf_con = worst_case_perf_con

    @property
    def found_violation(self):
        """
        bool : True if separation solution for at least one
        ``SeparationSolveCallResults`` object listed in self
        was reported to violate its corresponding performance
        constraint, False otherwise.
        """
        return any((solver_call_res.found_violation for solver_call_res in self.solver_call_results.values()))

    @property
    def violating_param_realization(self):
        """
        None or list of float : Uncertain parameter values for
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_perf_con``.
        If ``self.worst_case_perf_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_perf_con is not None:
            return self.solver_call_results[self.worst_case_perf_con].violating_param_realization
        else:
            return None

    @property
    def scaled_violations(self):
        """
        None or ComponentMap : Scaled performance constraint violations
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_perf_con``.
        If ``self.worst_case_perf_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_perf_con is not None:
            return self.solver_call_results[self.worst_case_perf_con].scaled_violations
        else:
            return None

    @property
    def violating_separation_variable_values(self):
        """
        None or ComponentMap : Second-stage and state variable values
        for maximally violating separation problem solution,
        specified according to solver call results object
        listed in self at index ``self.worst_case_perf_con``.
        If ``self.worst_case_perf_con`` is not specified,
        then None is returned.
        """
        if self.worst_case_perf_con is not None:
            return self.solver_call_results[self.worst_case_perf_con].variable_values
        else:
            return None

    @property
    def violated_performance_constraints(self):
        """
        list of Constraint : Performance constraints for which violation
        found.
        """
        return [con for con, solver_call_results in self.solver_call_results.items() if solver_call_results.found_violation]

    @property
    def subsolver_error(self):
        """
        bool : Return True if subsolver error reported for
        at least one ``SeparationSolveCallResults`` stored in
        `self`, False otherwise.
        """
        return any((solver_call_res.subsolver_error for solver_call_res in self.solver_call_results.values()))

    @property
    def time_out(self):
        """
        bool : Return True if time out reported for
        at least one ``SeparationSolveCallResults`` stored in
        `self`, False otherwise.
        """
        return any((solver_call_res.time_out for solver_call_res in self.solver_call_results.values()))

    def evaluate_total_solve_time(self, evaluator_func, **evaluator_func_kwargs):
        """
        Evaluate total time required by subordinate solvers
        for separation problem of interest.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolveResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by solvers.
        """
        return sum((res.evaluate_total_solve_time(evaluator_func) for res in self.solver_call_results.values()))