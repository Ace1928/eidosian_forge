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
def _transform_piecewise_linear_function(self, pw_linear_func, descend_into_expressions):
    if descend_into_expressions:
        return
    transBlock = self._get_transformation_block(pw_linear_func.parent_block())
    _functions = pw_linear_func.values() if pw_linear_func.is_indexed() else (pw_linear_func,)
    for pw_func in _functions:
        for pw_expr in pw_func._expressions.values():
            substitute_var = self._transform_pw_linear_expr(pw_expr.expr, pw_func, transBlock)
            pw_expr.expr = substitute_var
    pw_linear_func.deactivate()