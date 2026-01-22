import logging
from pyomo.common import deprecated
from pyomo.core.base import Transformation, TransformationFactory, Var, Suffix, Reals
@TransformationFactory.register('core.relax_discrete', doc='[DEPRECATED] Relax integer variables to continuous counterparts')
@deprecated('core.relax_discrete is deprecated.  Use core.relax_integer_vars', version='5.7')
class RelaxDiscreteVars(RelaxIntegerVars):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    def __init__(self, **kwds):
        super(RelaxDiscreteVars, self).__init__(**kwds)