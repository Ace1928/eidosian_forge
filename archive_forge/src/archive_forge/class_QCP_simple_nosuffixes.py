import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QCP_simple_nosuffixes(QCP_simple):
    description = 'QCP_simple_nosuffixes'
    test_pickling = False

    def __init__(self):
        QCP_simple.__init__(self)
        self.disable_suffix_tests = True
        self.add_results('QCP_simple.json')