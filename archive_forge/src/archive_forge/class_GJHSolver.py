import logging
import glob
from pyomo.common.tempfiles import TempfileManager
from pyomo.solvers.plugins.solvers.ASL import ASL
class GJHSolver(ASL):
    """
    An interface to the AMPL GJH "solver" for evaluating a model at a
    point.
    """

    def __init__(self, **kwds):
        kwds['type'] = 'gjh'
        kwds['symbolic_solver_labels'] = True
        super().__init__(**kwds)
        self.options.solver = 'gjh'
        self._metasolver = False

    def _initialize_callbacks(self, model):
        self._model = model
        self._model._gjh_info = None
        super()._initialize_callbacks(model)

    def _presolve(self, *args, **kwds):
        super()._presolve(*args, **kwds)
        self._gjh_file = self._soln_file[:-3] + 'gjh'
        TempfileManager.add_tempfile(self._gjh_file, exists=False)

    def _postsolve(self):
        self._model._gjh_info = readgjh(self._gjh_file)
        self._model = None
        return super()._postsolve()