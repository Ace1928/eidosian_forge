from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import DeveloperError
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.piecewise_to_mip_visitor import (
from pyomo.core import (
from pyomo.core.base import Transformation
from pyomo.core.base.block import _BlockData, Block
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import is_child_of
from pyomo.network import Port
def _get_transformation_block(self, parent):
    if parent in self._transformation_blocks:
        return self._transformation_blocks[parent]
    nm = unique_component_name(parent, '_pyomo_contrib_%s' % self._transformation_name)
    self._transformation_blocks[parent] = transBlock = Block()
    parent.add_component(nm, transBlock)
    transBlock.transformed_functions = Block(Any)
    return transBlock