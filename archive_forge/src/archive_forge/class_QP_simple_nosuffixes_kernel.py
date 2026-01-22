import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model
@register_model
class QP_simple_nosuffixes_kernel(QP_simple_kernel):
    description = 'QP_simple_nosuffixes'
    test_pickling = False

    def __init__(self):
        QP_simple.__init__(self)
        self.disable_suffix_tests = True
        self.add_results('QP_simple.json')