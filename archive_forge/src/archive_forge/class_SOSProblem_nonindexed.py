import math
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.opt import check_available_solvers
@unittest.skipIf(not solver_available, 'The solver is not available.')
class SOSProblem_nonindexed(object):
    """Test non-indexed SOS using a single pyomo Var component."""

    def verify(self, model, sos, exp_res, abs_tol, use_rule, case, show_output: bool=False):
        """Make sure the outcome is as expected."""
        opt = pyo.SolverFactory(solver_name)
        opt.solve(model, tee=show_output)
        assert len(model.mysos) != 0
        assert math.isclose(pyo.value(model.OBJ), exp_res, abs_tol=abs_tol)

    def do_it(self, test_number):
        sos, exp_res, abs_tol, use_rule, case = self.test_vectors[test_number]
        model = self.set_problem_up(case=case, n=sos, use_rule=use_rule)
        self.verify(model=model, sos=sos, exp_res=exp_res, abs_tol=abs_tol, use_rule=use_rule, case=case)
    test_vectors = [(1, 0.04999999999999999, 0.001, True, 0), (1, 0.04999999999999999, 0.001, False, 0), (2, -0.07500000000000001, 0.001, True, 0), (2, -0.07500000000000001, 0.001, False, 0), (1, 0.04999999999999999, 0.001, True, 1), (1, 0.04999999999999999, 0.001, False, 1), (2, -0.07500000000000001, 0.001, True, 1), (2, -0.07500000000000001, 0.001, False, 1), (1, 0.04999999999999999, 0.001, True, 2), (1, 0.04999999999999999, 0.001, False, 2), (2, -0.07500000000000001, 0.001, True, 2), (2, -0.07500000000000001, 0.001, False, 2), (1, 0.04999999999999999, 0.001, True, 3), (1, 0.04999999999999999, 0.001, False, 3), (2, -0.07500000000000001, 0.001, True, 3), (2, -0.07500000000000001, 0.001, False, 3), (1, 0.04999999999999999, 0.001, True, 4), (1, 0.04999999999999999, 0.001, False, 4), (2, -0.07500000000000001, 0.001, True, 4), (2, -0.07500000000000001, 0.001, False, 4), (1, 0.04999999999999999, 0.001, True, 5), (1, 0.04999999999999999, 0.001, False, 5), (2, -0.07500000000000001, 0.001, True, 5), (2, -0.07500000000000001, 0.001, False, 5), (1, 0.04999999999999999, 0.001, True, 6), (1, 0.04999999999999999, 0.001, False, 6), (2, -0.07500000000000001, 0.001, True, 6), (2, -0.07500000000000001, 0.001, False, 6), (1, 0.04999999999999999, 0.001, True, 7), (1, 0.04999999999999999, 0.001, False, 7), (2, -0.07500000000000001, 0.001, True, 7), (2, -0.07500000000000001, 0.001, False, 7)]

    def set_problem_up(self, case: int == 0, n: int=1, use_rule: bool=False):
        """Create the problem."""
        model = pyo.ConcreteModel()
        model.x = pyo.Var([1], domain=pyo.NonNegativeReals, bounds=(0, 40))
        model.A = pyo.Set(initialize=[1, 2, 4, 6])
        model.y = pyo.Var(model.A, domain=pyo.NonNegativeReals, bounds=(0, 2))
        model.OBJ = pyo.Objective(expr=1 * model.x[1] + 2 * model.y[1] + 3 * model.y[2] + -0.1 * model.y[4] + 0.5 * model.y[6])
        model.ConstraintYmin = pyo.Constraint(expr=model.x[1] + model.y[1] + model.y[2] + model.y[6] >= 0.25)
        if case == 0:
            if use_rule:

                def rule_mysos(m):
                    return [m.y[a] for a in m.A]
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, sos=n)
        elif case == 1:
            index = [2, 4, 6]
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in index], [i + 1 for i, _ in enumerate(index)])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, index=index, sos=n)
        elif case == 2:
            model.mysosindex = pyo.Set(initialize=[2, 4, 6], within=model.A)
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in m.mysosindex], [i + 1 for i, _ in enumerate(m.mysosindex)])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, index=model.mysosindex, sos=n)
        elif case == 3:
            index = [2, 4, 6]
            weights = {2: 25.0, 4: 18.0, 6: 22}
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in index], [weights[a] for a in index])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, index=index, weights=weights, sos=n)
        elif case == 4:
            model.mysosindex = pyo.Set(initialize=[2, 4, 6], within=model.A)
            model.mysosweights = pyo.Param(model.mysosindex, initialize={2: 25.0, 4: 18.0, 6: 22})
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in m.mysosindex], [m.mysosweights[a] for a in m.mysosindex])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, index=model.mysosindex, weights=model.mysosweights, sos=n)
        elif case == 5:
            weights = {1: 3, 2: 25.0, 4: 18.0, 6: 22}
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in m.y], [weights[a] for a in m.y])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, sos=n, weights=weights)
        elif case == 6:
            model.mysosweights = pyo.Param([1, 2, 4, 6], initialize={1: 3, 2: 25.0, 4: 18.0, 6: 22})
            if use_rule:

                def rule_mysos(m):
                    return ([m.y[a] for a in m.y], [m.mysosweights[a] for a in m.y])
                model.mysos = pyo.SOSConstraint(rule=rule_mysos, sos=n)
            else:
                model.mysos = pyo.SOSConstraint(var=model.y, sos=n, weights=model.mysosweights)
        else:
            raise NotImplementedError
        return model