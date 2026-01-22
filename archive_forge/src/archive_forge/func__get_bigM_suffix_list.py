from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _get_bigM_suffix_list(block, stopping_block=None):
    suffix_list = []
    while block is not None:
        bigm = block.component('BigM')
        if type(bigm) is Suffix:
            suffix_list.append(bigm)
        if block is stopping_block:
            break
        block = block.parent_block()
    return suffix_list