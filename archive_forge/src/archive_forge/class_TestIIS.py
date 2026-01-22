import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.iis import write_iis
from pyomo.contrib.iis.iis import _supported_solvers
from pyomo.common.tempfiles import TempfileManager
import os
class TestIIS(unittest.TestCase):

    @unittest.skipUnless(pyo.SolverFactory('cplex_persistent').available(exception_flag=False), 'CPLEX not available')
    def test_write_iis_cplex(self):
        _test_iis('cplex')

    @unittest.skipUnless(pyo.SolverFactory('gurobi_persistent').available(exception_flag=False), 'Gurobi not available')
    def test_write_iis_gurobi(self):
        _test_iis('gurobi')

    @unittest.skipUnless(pyo.SolverFactory('xpress_persistent').available(exception_flag=False), 'Xpress not available')
    def test_write_iis_xpress(self):
        _test_iis('xpress')

    @unittest.skipUnless(pyo.SolverFactory('cplex_persistent').available(exception_flag=False) or pyo.SolverFactory('gurobi_persistent').available(exception_flag=False) or pyo.SolverFactory('xpress_persistent').available(exception_flag=False), 'Persistent solver not available')
    def test_write_iis_any_solver(self):
        _test_iis(None)

    @unittest.skipIf(pyo.SolverFactory('cplex_persistent').available(exception_flag=False), 'CPLEX available')
    def test_exception_cplex_not_available(self):
        self._assert_raises_unavailable_solver('cplex')

    @unittest.skipIf(pyo.SolverFactory('gurobi_persistent').available(exception_flag=False), 'Gurobi available')
    def test_exception_gurobi_not_available(self):
        self._assert_raises_unavailable_solver('gurobi')

    @unittest.skipIf(pyo.SolverFactory('xpress_persistent').available(exception_flag=False), 'Xpress available')
    def test_exception_xpress_not_available(self):
        self._assert_raises_unavailable_solver('xpress')

    @unittest.skipIf(pyo.SolverFactory('cplex_persistent').available(exception_flag=False) or pyo.SolverFactory('gurobi_persistent').available(exception_flag=False) or pyo.SolverFactory('xpress_persistent').available(exception_flag=False), 'Persistent solver available')
    def test_exception_iis_no_solver_available(self):
        with self.assertRaises(RuntimeError, msg=f'Could not find a solver to use, supported solvers are {_supported_solvers}'):
            _test_iis(None)

    def _assert_raises_unavailable_solver(self, solver_name):
        with self.assertRaises(RuntimeError, msg=f'The Pyomo persistent interface to {solver_name} could not be found.'):
            _test_iis(solver_name)