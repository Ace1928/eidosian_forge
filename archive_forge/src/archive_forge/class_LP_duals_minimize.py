import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_duals_minimize(_BaseTestModel):
    """
    A continuous linear model designed to test every form of
    constraint when collecting duals for a minimization
    objective
    """
    description = 'LP_duals_minimize'
    level = ('nightly', 'expensive')
    capabilities = set(['linear'])
    size = (12, 12, None)

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = None
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.s = RangeSet(1, 12)
        model.x = Var(model.s)
        model.x[1].setlb(-1)
        model.x[1].setub(1)
        model.x[2].setlb(-1)
        model.x[2].setub(1)
        model.obj = Objective(expr=sum((model.x[i] * (-1) ** (i + 1) for i in model.x.index_set())))
        model.c = ConstraintList()
        model.c.add(Constraint.Skip)
        model.c.add(Constraint.Skip)
        model.c.add(model.x[3] >= -1.0)
        model.c.add(model.x[4] <= 1.0)
        model.c.add(model.x[5] == -1.0)
        model.c.add(model.x[6] == -1.0)
        model.c.add(model.x[7] == 1.0)
        model.c.add(model.x[8] == 1.0)
        model.c.add((-1.0, model.x[9], -1.0))
        model.c.add((-1.0, model.x[10], -1.0))
        model.c.add((1.0, model.x[11], 1.0))
        model.c.add((1.0, model.x[12], 1.0))
        model.c_inactive = ConstraintList()
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(Constraint.Skip)
        model.c_inactive.add(model.x[3] >= -2.0)
        model.c_inactive.add(model.x[4] <= 2.0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        for i in model.s:
            model.x[i].value = None