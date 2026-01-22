import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
@unittest.skipIf(not solver_available, 'The solver is not available.')
class SOSProblem_indexedmultivar(object):
    """Test indexed SOS made up of different Var components."""

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
    test_vectors = [(1, -0.075, 0.001), (2, 1.1, 0.001)]

    def set_problem_up(self, n: int=1):
        """Create the problem."""
        model = pyo.AbstractModel()
        model.E = pyo.Set(initialize=[1, 2])
        model.A = pyo.Set(initialize=[1, 2, 3, 5, 6])
        model.B = pyo.Set(initialize=[2, 4])
        model.x = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 40))
        model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals)
        model.param_cx = pyo.Param(model.E, initialize={1: 1, 2: 1.5})
        model.param_cy = pyo.Param(model.A, initialize={1: 2, 2: 3, 3: -0.1, 5: 0.5, 6: 4})

        def obj_f(m):
            return sum((m.param_cx[e] * m.x[e] for e in m.E)) + sum((m.param_cy[a] * m.y[a] for a in m.A))
        model.OBJ = pyo.Objective(rule=obj_f)

        def constr_ya_lb(m, a):
            return m.y[a] <= 2
        model.ConstraintYa_lb = pyo.Constraint(model.A, rule=constr_ya_lb)

        def constr_y_lb(m):
            return m.x[1] + m.x[2] + m.y[1] + m.y[2] + m.y[5] + m.y[6] >= 0.25
        model.ConstraintY_lb = pyo.Constraint(rule=constr_y_lb)
        if n == 2:

            def constr_y2_lb(m):
                return m.y[2] + m.y[5] + m.y[6] >= 2.1
            model.ConstraintY2_lb = pyo.Constraint(rule=constr_y2_lb)
        model.mysosindex_x = pyo.Set(model.B, initialize={2: [1], 4: [2]})
        model.mysosindex_y = pyo.Set(model.B, initialize={2: [1, 3], 4: [2, 5, 6]})
        model.mysosweights_x = pyo.Param(model.E, initialize={1: 4, 2: 8})
        model.mysosweights_y = pyo.Param(model.A, initialize={1: 25.0, 3: 18.0, 2: 3, 5: 7, 6: 10})

        def rule_mysos(m, b):
            var_list = [m.x[e] for e in m.mysosindex_x[b]]
            var_list.extend([m.y[a] for a in m.mysosindex_y[b]])
            weight_list = [m.mysosweights_x[e] for e in m.mysosindex_x[b]]
            weight_list.extend([m.mysosweights_y[a] for a in m.mysosindex_y[b]])
            return (var_list, weight_list)
        model.mysos = pyo.SOSConstraint(model.B, rule=rule_mysos, sos=n)
        return model