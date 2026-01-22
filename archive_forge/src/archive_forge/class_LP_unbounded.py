import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class LP_unbounded(_BaseTestModel):
    """
    A unbounded linear program
    """
    description = 'LP_unbounded'
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.solve_should_fail = True
        self.add_results(self.description + '.json')

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description
        model.x = Var()
        model.y = Var()
        model.o = Objective(expr=model.x + model.y)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            assert results['Solver'][0]['termination condition'] in (TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded)
        else:
            tester.assertIn(results['Solver'][0]['termination condition'], (TerminationCondition.unbounded, TerminationCondition.infeasibleOrUnbounded))