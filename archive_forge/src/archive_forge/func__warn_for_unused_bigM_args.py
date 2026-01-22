from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _warn_for_unused_bigM_args(bigM, used_args, logger):
    if bigM is not None:
        unused_args = ComponentSet(bigM.keys()) - ComponentSet(used_args.keys())
        if len(unused_args) > 0:
            warning_msg = 'Unused arguments in the bigM map! These arguments were not used by the transformation:\n'
            for component in unused_args:
                if isinstance(component, (tuple, list)) and len(component) == 2:
                    warning_msg += '\t(%s, %s)\n' % (component[0].name, component[1].name)
                elif hasattr(component, 'name'):
                    warning_msg += '\t%s\n' % component.name
                else:
                    warning_msg += '\t%s\n' % component
            logger.warning(warning_msg)