from pyomo.common import deprecated
from pyomo.core.base import TransformationFactory
from pyomo.core.plugins.transform.discrete_vars import RelaxIntegerVars
@TransformationFactory.register('core.relax_integrality', doc='[DEPRECATED] Create a model where integer variables are replaced with real variables.')
@deprecated('core.relax_integrality is deprecated.  Use core.relax_integer_vars', version='5.7')
class RelaxIntegrality(RelaxIntegerVars):
    """
    This plugin relaxes integrality in a Pyomo model.
    """

    def __init__(self, **kwds):
        super(RelaxIntegrality, self).__init__(**kwds)