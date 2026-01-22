from pyomo.environ import (
from pyomo.common.collections import ComponentMap
class SimpleMINLP(ConcreteModel):
    """Example 1 Outer Approximation and Extended Cutting Planes."""

    def __init__(self, *args, **kwargs):
        """Create the problem."""
        kwargs.setdefault('name', 'SimpleMINLP2')
        super(SimpleMINLP, self).__init__(*args, **kwargs)
        m = self
        'Set declarations'
        I = m.I = RangeSet(1, 4, doc='continuous variables')
        J = m.J = RangeSet(1, 3, doc='discrete variables')
        initY = {1: 1, 2: 0, 3: 1}
        initX = {1: 0, 2: 0, 3: 0, 4: 0}
        'Variable declarations'
        Y = m.Y = Var(J, domain=Binary, initialize=initY)
        X = m.X = Var(I, domain=NonNegativeReals, initialize=initX, bounds=(0, 2))
        'Constraint definitions'
        m.const1 = Constraint(expr=0.8 * log(X[2] + 1) + 0.96 * log(X[1] - X[2] + 1) - 0.8 * X[3] >= 0)
        m.const2 = Constraint(expr=log(X[2] + 1) + 1.2 * log(X[1] - X[2] + 1) - X[3] - 2 * Y[3] >= -2)
        m.const3 = Constraint(expr=10 * X[1] - 7 * X[3] - 18 * log(X[2] + 1) - 19.2 * log(X[1] - X[2] + 1) + 10 - X[4] <= 0)
        m.const4 = Constraint(expr=X[2] - X[1] <= 0)
        m.const5 = Constraint(expr=X[2] - 2 * Y[1] <= 0)
        m.const6 = Constraint(expr=X[1] - X[2] - 2 * Y[2] <= 0)
        m.const7 = Constraint(expr=Y[1] + Y[2] <= 1)
        'Cost (objective) function definition'
        m.objective = Objective(expr=+5 * Y[1] + 6 * Y[2] + 8 * Y[3] + X[4], sense=minimize)
        'Bound definitions'
        x_ubs = {1: 2, 2: 2, 3: 1, 4: 100}
        for i, x_ub in x_ubs.items():
            X[i].setub(x_ub)
        m.optimal_value = 6.00976
        m.optimal_solution = ComponentMap()
        m.optimal_solution[m.X[1]] = 1.3009758908698426
        m.optimal_solution[m.X[2]] = 0.0
        m.optimal_solution[m.X[3]] = 1.0
        m.optimal_solution[m.X[4]] = 0.009758908698423729
        m.optimal_solution[m.Y[1]] = 0.0
        m.optimal_solution[m.Y[2]] = 1.0
        m.optimal_solution[m.Y[3]] = 0.0