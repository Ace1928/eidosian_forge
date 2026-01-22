import io
import logging
import sys
from collections.abc import Sequence
from typing import Optional, List, TextIO
from pyomo.common.config import (
from pyomo.common.log import LogStream
from pyomo.common.numeric_types import native_logical_types
from pyomo.common.timing import HierarchicalTimer
class SolverConfig(ConfigDict):
    """
    Base config for all direct solver interfaces
    """

    def __init__(self, description=None, doc=None, implicit=False, implicit_domain=None, visibility=0):
        super().__init__(description=description, doc=doc, implicit=implicit, implicit_domain=implicit_domain, visibility=visibility)
        self.tee: List[TextIO] = self.declare('tee', ConfigValue(domain=TextIO_or_Logger, default=False, description='``tee`` accepts :py:class:`bool`,\n                :py:class:`io.TextIOBase`, or :py:class:`logging.Logger`\n                (or a list of these types).  ``True`` is mapped to\n                ``sys.stdout``.  The solver log will be printed to each of\n                these streams / destinations.'))
        self.working_dir: Optional[Path] = self.declare('working_dir', ConfigValue(domain=Path(), default=None, description='The directory in which generated files should be saved. This replaces the `keepfiles` option.'))
        self.load_solutions: bool = self.declare('load_solutions', ConfigValue(domain=Bool, default=True, description='If True, the values of the primal variables will be loaded into the model.'))
        self.raise_exception_on_nonoptimal_result: bool = self.declare('raise_exception_on_nonoptimal_result', ConfigValue(domain=Bool, default=True, description='If False, the `solve` method will continue processing even if the returned result is nonoptimal.'))
        self.symbolic_solver_labels: bool = self.declare('symbolic_solver_labels', ConfigValue(domain=Bool, default=False, description='If True, the names given to the solver will reflect the names of the Pyomo components. Cannot be changed after set_instance is called.'))
        self.timer: Optional[HierarchicalTimer] = self.declare('timer', ConfigValue(default=None, description='A timer object for recording relevant process timing data.'))
        self.threads: Optional[int] = self.declare('threads', ConfigValue(domain=NonNegativeInt, description='Number of threads to be used by a solver.', default=None))
        self.time_limit: Optional[float] = self.declare('time_limit', ConfigValue(domain=NonNegativeFloat, description='Time limit applied to the solver (in seconds).'))
        self.solver_options: ConfigDict = self.declare('solver_options', ConfigDict(implicit=True, description='Options to pass to the solver.'))