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
def _get_legendre_constants(self, currentds):
    """
        This function sets the legendre collocation points and a values
        depending on how many collocation points have been specified and
        whether or not the user has numpy
        """
    if not numpy_available:
        if self._ncp[currentds] > 10:
            raise ValueError('Numpy was not found so the maximum number of collocation points is 10')
        from pyomo.dae.utilities import legendre_tau_dict, legendre_adot_dict, legendre_adotdot_dict, legendre_afinal_dict
        self._tau[currentds] = legendre_tau_dict[self._ncp[currentds]]
        self._adot[currentds] = legendre_adot_dict[self._ncp[currentds]]
        self._adotdot[currentds] = legendre_adotdot_dict[self._ncp[currentds]]
        self._afinal[currentds] = legendre_afinal_dict[self._ncp[currentds]]
    else:
        alpha = 0
        beta = 0
        k = self._ncp[currentds]
        cp = calc_cp(alpha, beta, k)
        cp.insert(0, 0.0)
        adot = calc_adot(cp, 1)
        adotdot = calc_adot(cp, 2)
        afinal = calc_afinal(cp)
        self._tau[currentds] = cp
        self._adot[currentds] = adot
        self._adotdot[currentds] = adotdot
        self._afinal[currentds] = afinal