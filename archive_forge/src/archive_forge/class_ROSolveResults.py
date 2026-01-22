class ROSolveResults(object):
    """
    PyROS solver results object.

    Parameters
    ----------
    config : ConfigDict, optional
        User-specified solver settings.
    iterations : int, optional
        Number of iterations required.
    time : float, optional
        Total elapsed time (or wall time), in seconds.
    final_objective_value : float, optional
        Final objective function value to report.
    pyros_termination_condition : pyrosTerminationCondition, optional
        PyROS-specific termination condition.

    Attributes
    ----------
    config : ConfigDict, optional
        User-specified solver settings.
    iterations : int, optional
        Number of iterations required by PyROS.
    time : float, optional
        Total elapsed time (or wall time), in seconds.
    final_objective_value : float, optional
        Final objective function value to report.
    pyros_termination_condition : pyros.util.pyrosTerminationStatus
        Indicator of the manner of termination.
    """

    def __init__(self, config=None, iterations=None, time=None, final_objective_value=None, pyros_termination_condition=None):
        """Initialize self (see class docstring)."""
        self.config = config
        self.iterations = iterations
        self.time = time
        self.final_objective_value = final_objective_value
        self.pyros_termination_condition = pyros_termination_condition

    def __str__(self):
        """
        Generate string representation of self.
        Does not include any information about `self.config`.
        """
        lines = ['Termination stats:']
        attr_name_format_dict = {'iterations': ('Iterations', "f'{val}'"), 'time': ('Solve time (wall s)', "f'{val:.3f}'"), 'final_objective_value': ('Final objective value', "f'{val:.4e}'"), 'pyros_termination_condition': ('Termination condition', "f'{val}'")}
        attr_desc_pad_length = max((len(desc) for desc, _ in attr_name_format_dict.values()))
        for attr_name, (attr_desc, fmt_str) in attr_name_format_dict.items():
            val = getattr(self, attr_name)
            val_str = eval(fmt_str) if val is not None else str(val)
            lines.append(f' {attr_desc:<{attr_desc_pad_length}s} : {val_str}')
        return '\n'.join(lines)