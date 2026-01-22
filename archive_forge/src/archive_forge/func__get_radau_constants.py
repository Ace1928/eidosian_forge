import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def _get_radau_constants(self, currentds):
    """
        This function sets the radau collocation points and a values depending
        on how many collocation points have been specified and whether or not
        the user has numpy
        """
    if not numpy_available:
        if self._ncp[currentds] > 10:
            raise ValueError('Numpy was not found so the maximum number of collocation points is 10')
        from pyomo.dae.utilities import radau_tau_dict, radau_adot_dict, radau_adotdot_dict
        self._tau[currentds] = radau_tau_dict[self._ncp[currentds]]
        self._adot[currentds] = radau_adot_dict[self._ncp[currentds]]
        self._adotdot[currentds] = radau_adotdot_dict[self._ncp[currentds]]
        self._afinal[currentds] = None
    else:
        alpha = 1
        beta = 0
        k = self._ncp[currentds] - 1
        cp = calc_cp(alpha, beta, k)
        cp.insert(0, 0.0)
        cp.append(1.0)
        adot = calc_adot(cp, 1)
        adotdot = calc_adot(cp, 2)
        self._tau[currentds] = cp
        self._adot[currentds] = adot
        self._adotdot[currentds] = adotdot
        self._afinal[currentds] = None