import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_variable_data(self):
    solver_model = cplex.Cplex()
    var_data = _VariableData(solver_model)
    var_data.add(lb=0, ub=1, type_=solver_model.variables.type.binary, name='var1')
    var_data.add(lb=0, ub=10, type_=solver_model.variables.type.integer, name='var2')
    var_data.add(lb=-cplex.infinity, ub=cplex.infinity, type_=solver_model.variables.type.continuous, name='var3')
    self.assertEqual(solver_model.variables.get_num(), 0)
    var_data.store_in_cplex()
    self.assertEqual(solver_model.variables.get_num(), 3)