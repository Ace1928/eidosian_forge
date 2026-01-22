from pyomo.common.deprecation import RenamedClass
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.indexed_component import rule_wrapper
from pyomo.core.base.expression import (
from pyomo.dae.contset import ContinuousSet
from pyomo.dae.diffvar import DAE_Error
class ScalarIntegral(ScalarExpression, Integral):
    """
    An integral that will have no indexing sets after applying a numerical
    integration transformation
    """

    def __init__(self, *args, **kwds):
        _GeneralExpressionData.__init__(self, None, component=self)
        Integral.__init__(self, *args, **kwds)

    def clear(self):
        self._data = {}

    def is_fully_discretized(self):
        """
        Checks to see if all ContinuousSets indexing this Integral have been
        discretized
        """
        if 'scheme' not in self._wrt.get_discretization_info():
            return False
        return True