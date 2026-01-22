from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QP_constant_objective(_BaseTestModel):
    """
    A continuous linear model with a constant objective that starts
    as quadratic expression
    """
    description = 'QP_constant_objective'
    capabilities = set(['linear', 'quadratic_objective'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x ** 2 - model.x ** 2)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 1.0