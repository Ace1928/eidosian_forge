import logging
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def _forward_transform(v, s):
    """
    Applies the Forward Difference formula of order O(h) for first derivatives
    """

    def _fwd_fun(i):
        tmp = list(s)
        idx = s.ord(i) - 1
        return 1 / (tmp[idx + 1] - tmp[idx]) * (v(tmp[idx + 1]) - v(tmp[idx]))
    return _fwd_fun