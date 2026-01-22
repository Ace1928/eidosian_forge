class SeparationResults:
    """
    Container for results of PyROS separation problem routine.

    Parameters
    ----------
    local_separation_loop_results : None or SeparationLoopResults
        Local separation problem loop results.
    global_separation_loop_results : None or SeparationLoopResults
        Global separation problem loop results.

    Attributes
    ----------
    local_separation_loop_results
    global_separation_loop_results
    main_loop_results
    subsolver_error
    time_out
    solved_locally
    solved_globally
    found_violation
    violating_param_realization
    scaled_violations
    violating_separation_variable_values
    robustness_certified
    """

    def __init__(self, local_separation_loop_results, global_separation_loop_results):
        """Initialize self (see class docstring)."""
        self.local_separation_loop_results = local_separation_loop_results
        self.global_separation_loop_results = global_separation_loop_results

    @property
    def time_out(self):
        """
        bool : True if time out found for local or global
        separation loop, False otherwise.
        """
        local_time_out = self.solved_locally and self.local_separation_loop_results.time_out
        global_time_out = self.solved_globally and self.global_separation_loop_results.time_out
        return local_time_out or global_time_out

    @property
    def subsolver_error(self):
        """
        bool : True if subsolver error found for local or global
        separation loop, False otherwise.
        """
        local_subsolver_error = self.solved_locally and self.local_separation_loop_results.subsolver_error
        global_subsolver_error = self.solved_globally and self.global_separation_loop_results.subsolver_error
        return local_subsolver_error or global_subsolver_error

    @property
    def solved_locally(self):
        """
        bool : true if local separation loop was invoked,
        False otherwise.
        """
        return self.local_separation_loop_results is not None

    @property
    def solved_globally(self):
        """
        bool : True if global separation loop was invoked,
        False otherwise.
        """
        return self.global_separation_loop_results is not None

    def get_violating_attr(self, attr_name):
        """
        If separation problems solved globally, returns
        value of attribute of global separation loop results.

        Otherwise, if separation problems solved locally,
        returns value of attribute of local separation loop results.
        If local separation loop results specified, return
        value of attribute of local separation loop results.

        Otherwise, if global separation loop results specified,
        return value of attribute of global separation loop
        results.

        Otherwise, return None.

        Parameters
        ----------
        attr_name : str
            Name of attribute to be retrieved. Should be
            valid attribute name for object of type
            ``SeparationLoopResults``.

        Returns
        -------
        object
            Attribute value.
        """
        return getattr(self.main_loop_results, attr_name, None)

    @property
    def worst_case_perf_con(self):
        """
        ConstraintData : Performance constraint corresponding to the
        separation solution chosen for the next master problem.
        """
        return self.get_violating_attr('worst_case_perf_con')

    @property
    def main_loop_results(self):
        """
        SeparationLoopResults : Main separation loop results.
        In particular, this is considered to be the global
        loop result if solved globally, and the local loop
        results otherwise.
        """
        if self.solved_globally:
            return self.global_separation_loop_results
        return self.local_separation_loop_results

    @property
    def found_violation(self):
        """
        bool : True if ``found_violation`` attribute for
        main separation loop results is True, False otherwise.
        """
        found_viol = self.get_violating_attr('found_violation')
        if found_viol is None:
            found_viol = False
        return found_viol

    @property
    def violating_param_realization(self):
        """
        None or list of float : Uncertain parameter values
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_perf_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr('violating_param_realization')

    @property
    def scaled_violations(self):
        """
        None or ComponentMap : Scaled performance constraint violations
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_perf_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr('scaled_violations')

    @property
    def violating_separation_variable_values(self):
        """
        None or ComponentMap : Second-stage and state variable values
        for maximally violating separation problem solution
        reported in local or global separation loop results.
        If no such solution found, (i.e. ``worst_case_perf_con``
        set to None for both local and global loop results),
        then None is returned.
        """
        return self.get_violating_attr('violating_separation_variable_values')

    @property
    def violated_performance_constraints(self):
        """
        Return list of violated performance constraints.
        """
        return self.get_violating_attr('violated_performance_constraints')

    def evaluate_local_solve_time(self, evaluator_func, **evaluator_func_kwargs):
        """
        Evaluate total time required by local subordinate solvers
        for separation problem of interest.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolverResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by local solvers.
        """
        if self.solved_locally:
            return self.local_separation_loop_results.evaluate_total_solve_time(evaluator_func, **evaluator_func_kwargs)
        else:
            return 0

    def evaluate_global_solve_time(self, evaluator_func, **evaluator_func_kwargs):
        """
        Evaluate total time required by global subordinate solvers
        for separation problem of interest.

        Parameters
        ----------
        evaluator_func : callable
            Solve time evaluator function.
            This callable should accept an object of type
            ``pyomo.opt.results.SolverResults``, and
            return a float equal to the time required.
        **evaluator_func_kwargs : dict, optional
            Keyword arguments to evaluator function.

        Returns
        -------
        float
            Total time spent by global solvers.
        """
        if self.solved_globally:
            return self.global_separation_loop_results.evaluate_total_solve_time(evaluator_func, **evaluator_func_kwargs)
        else:
            return 0

    @property
    def robustness_certified(self):
        """
        bool : Return True if separation results certify that
        first-stage solution is robust, False otherwise.
        """
        assert self.solved_locally or self.solved_globally
        if self.time_out or self.subsolver_error:
            return False
        if self.solved_locally:
            heuristically_robust = not self.local_separation_loop_results.found_violation
        else:
            heuristically_robust = None
        if self.solved_globally:
            is_robust = not self.global_separation_loop_results.found_violation
        else:
            is_robust = heuristically_robust
        return is_robust

    def generate_subsolver_results(self, include_local=True, include_global=True):
        """
        Generate flattened sequence all Pyomo SolverResults objects
        for all ``SeparationSolveCallResults`` objects listed in
        the local and global ``SeparationLoopResults``
        attributes of `self`.

        Yields
        ------
        pyomo.opt.SolverResults
        """
        if include_local and self.local_separation_loop_results is not None:
            all_local_call_results = self.local_separation_loop_results.solver_call_results.values()
            for solve_call_res in all_local_call_results:
                for res in solve_call_res.results_list:
                    yield res
        if include_global and self.global_separation_loop_results is not None:
            all_global_call_results = self.global_separation_loop_results.solver_call_results.values()
            for solve_call_res in all_global_call_results:
                for res in solve_call_res.results_list:
                    yield res