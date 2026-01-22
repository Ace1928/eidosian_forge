import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
@unittest.skipIf(not solver_available, 'The solver is not available.')
class SOSProblem_nonindexed_multivar(object):
    """Test non-indexed SOS made up of different Var components."""

    def verify(self, model, sos, exp_res, abs_tol, show_output: bool=False):
        """Make sure the outcome is as expected."""
        opt = pyo.SolverFactory(solver_name)
        problem = model.create_instance()
        opt.solve(problem, tee=show_output)
        assert len(problem.mysos) != 0
        assert math.isclose(pyo.value(problem.OBJ), exp_res, abs_tol=abs_tol)

    def do_it(self, test_number):
        sos, exp_res, abs_tol = self.test_vectors[test_number]
        model = self.set_problem_up(n=sos)
        self.verify(model=model, sos=sos, exp_res=exp_res, abs_tol=abs_tol)
    test_vectors = [(1, 0.125, 0.001), (2, -0.07500000000000001, 0.001)]

    def set_problem_up(self, n: int=1):
        """Create the problem."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0, 40))
        model.A = pyo.Set(initialize=[1, 2, 4, 6])
        model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals, bounds=(0, 2))
        model.OBJ = pyo.Objective(expr=1 * model.x[1] + 2 * model.y[1] + 3 * model.y[2] + -0.1 * model.y[4] + 0.5 * model.y[6])
        model.ConstraintYmin = pyo.Constraint(expr=model.x[1] + model.y[1] + model.y[2] + model.y[6] >= 0.25)

        def rule_mysos(m):
            var_list = [m.x[a] for a in m.x]
            var_list.extend([m.y[a] for a in m.A])
            weight_list = [i + 1 for i in range(len(var_list))]
            return (var_list, weight_list)
        model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
        return model