import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
@unittest.skipIf(not cplexpy_available, "The 'cplex' python bindings are not available")
class TestAddVar(unittest.TestCase):

    def test_add_single_variable(self):
        """Test that the variable is added correctly to `solver_model`."""
        model = ConcreteModel()
        opt = SolverFactory('cplex', solver_io='python')
        opt._set_instance(model)
        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)
        model.X = Var(within=Binary)
        var_interface = opt._solver_model.variables
        with unittest.mock.patch.object(var_interface, 'add', wraps=var_interface.add) as wrapped_add_call, unittest.mock.patch.object(var_interface, 'set_lower_bounds', wraps=var_interface.set_lower_bounds) as wrapped_lb_call, unittest.mock.patch.object(var_interface, 'set_upper_bounds', wraps=var_interface.set_upper_bounds) as wrapped_ub_call:
            opt._add_var(model.X)
            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(wrapped_add_call.call_args, ({'lb': [0], 'names': ['x1'], 'types': ['B'], 'ub': [1]},))
            self.assertFalse(wrapped_lb_call.called)
            self.assertFalse(wrapped_ub_call.called)
        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_single_variable(self):
        """Test that the variable is added correctly to `solver_model`."""
        model = ConcreteModel()
        opt = SolverFactory('cplex', solver_io='python')
        opt._set_instance(model)
        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)
        model.X = Var(within=Binary)
        with unittest.mock.patch.object(opt._solver_model.variables, 'add', wraps=opt._solver_model.variables.add) as wrapped_add_call:
            opt._add_block(model)
            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(wrapped_add_call.call_args, ({'lb': [0], 'names': ['x1'], 'types': ['B'], 'ub': [1]},))
        self.assertEqual(opt._solver_model.variables.get_num(), 1)
        self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)

    def test_add_block_containing_multiple_variables(self):
        """Test that:
        - The variable is added correctly to `solver_model`
        - The CPLEX `variables` interface is called only once
        - Fixed variable bounds are set correctly
        """
        model = ConcreteModel()
        opt = SolverFactory('cplex', solver_io='python')
        opt._set_instance(model)
        self.assertEqual(opt._solver_model.variables.get_num(), 0)
        model.X1 = Var(within=Binary)
        model.X2 = Var(within=NonNegativeReals)
        model.X3 = Var(within=NonNegativeIntegers)
        model.X3.fix(5)
        with unittest.mock.patch.object(opt._solver_model.variables, 'add', wraps=opt._solver_model.variables.add) as wrapped_add_call:
            opt._add_block(model)
            self.assertEqual(wrapped_add_call.call_count, 1)
            self.assertEqual(wrapped_add_call.call_args, ({'lb': [0, 0, 5], 'names': ['x1', 'x2', 'x3'], 'types': ['B', 'C', 'I'], 'ub': [1, cplex.infinity, 5]},))
        self.assertEqual(opt._solver_model.variables.get_num(), 3)