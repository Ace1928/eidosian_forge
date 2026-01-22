from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _get_linear_approximation_expr(normal_vec, point):
    """Returns constraint linearly approximating constraint normal to normal_vec
    at point"""
    body = 0
    for coef, v in zip(point, normal_vec):
        body -= coef * v
    return body >= -sum((normal_vec[idx] * v.value for idx, v in enumerate(point)))