import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_constant_objective2(_BaseTestModel):
    """
    A continuous linear model with a constant objective that
    starts as a linear expression
    """
    description = 'LP_constant_objective2'
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x - model.x)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 1.0