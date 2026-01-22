import re
import sys
import time
import logging
import shlex
from pyomo.common import Factory
from pyomo.common.errors import ApplicationError
from pyomo.common.collections import Bunch
from pyomo.opt.base.convert import convert_problem
from pyomo.opt.base.formats import ResultsFormat
import pyomo.opt.base.results
class UnknownSolver(object):

    def __init__(self, *args, **kwds):
        if 'type' in kwds:
            self.type = kwds['type']
        else:
            raise ValueError("Expected option 'type' for UnknownSolver constructor")
        self.options = {}
        self._args = args
        self._kwds = kwds

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def available(self, exception_flag=True):
        """Determine if this optimizer is available."""
        if exception_flag:
            raise ApplicationError('Solver (%s) not available' % str(self.name))
        return False

    def license_is_valid(self):
        """True if the solver is present and has a valid license (if applicable)"""
        return False

    def warm_start_capable(self):
        """True is the solver can accept a warm-start solution."""
        return False

    def solve(self, *args, **kwds):
        """Perform optimization and return an SolverResults object."""
        self._solver_error('solve')

    def reset(self):
        """Reset the state of an optimizer"""
        self._solver_error('reset')

    def set_options(self, istr):
        """Set the options in the optimizer from a string."""
        self._solver_error('set_options')

    def __bool__(self):
        return self.available()

    def __getattr__(self, attr):
        self._solver_error(attr)

    def _solver_error(self, method_name):
        raise RuntimeError('Attempting to use an unavailable solver.\n\nThe SolverFactory was unable to create the solver "%s"\nand returned an UnknownSolver object.  This error is raised at the point\nwhere the UnknownSolver object was used as if it were valid (by calling\nmethod "%s").\n\nThe original solver was created with the following parameters:\n\t' % (self.type, method_name) + '\n\t'.join(('%s: %s' % i for i in sorted(self._kwds.items()))) + '\n\t_args: %s' % (self._args,) + '\n\toptions: %s' % (self.options,))