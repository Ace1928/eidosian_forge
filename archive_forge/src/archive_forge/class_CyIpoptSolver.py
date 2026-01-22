import io
import sys
import logging
import os
import abc
from pyomo.common.deprecation import relocated_module_attribute
from pyomo.common.dependencies import attempt_import, numpy as np, numpy_available
from pyomo.common.tee import redirect_fd, TeeStream
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.common.timing import TicTocTimer
from pyomo.core.base import Block, Objective, minimize
from pyomo.opt import SolverStatus, SolverResults, TerminationCondition, ProblemSense
from pyomo.opt.results.solution import Solution
class CyIpoptSolver(object):

    def __init__(self, problem_interface, options=None):
        """Create an instance of the CyIpoptSolver. You must
        provide a problem_interface that corresponds to
        the abstract class CyIpoptProblemInterface
        options can be provided as a dictionary of key value
        pairs
        """
        self._problem = problem_interface
        self._options = options
        if options is not None:
            assert isinstance(options, dict)
        else:
            self._options = dict()

    def solve(self, x0=None, tee=False):
        if x0 is None:
            x0 = self._problem.x_init()
        xstart = x0
        cyipopt_solver = self._problem
        obj_scaling, x_scaling, g_scaling = self._problem.scaling_factors()
        if any((_ is not None for _ in (obj_scaling, x_scaling, g_scaling))):
            if obj_scaling is None:
                obj_scaling = 1.0
            if x_scaling is None:
                x_scaling = np.ones(nx)
            if g_scaling is None:
                g_scaling = np.ones(ng)
            try:
                set_scaling = cyipopt_solver.set_problem_scaling
            except AttributeError:
                set_scaling = cyipopt_solver.setProblemScaling
            set_scaling(obj_scaling, x_scaling, g_scaling)
        try:
            add_option = cyipopt_solver.add_option
        except AttributeError:
            add_option = cyipopt_solver.addOption
        for k, v in self._options.items():
            add_option(k, v)
        with TeeStream(sys.stdout) as _teeStream:
            if tee:
                try:
                    fd = sys.stdout.fileno()
                except (io.UnsupportedOperation, AttributeError):
                    fd = _teeStream.STDOUT.fileno()
            else:
                fd = None
            with redirect_fd(fd=1, output=fd, synchronize=False):
                x, info = cyipopt_solver.solve(xstart)
        return (x, info)