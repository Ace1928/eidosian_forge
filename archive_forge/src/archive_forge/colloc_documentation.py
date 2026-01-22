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

        This method will add additional constraints to a model to reduce the
        number of free collocation points (degrees of freedom) for a particular
        variable.

        Parameters
        ----------
        instance : Pyomo model
            The discretized Pyomo model to add constraints to

        var : ``pyomo.environ.Var``
            The Pyomo variable for which the degrees of freedom will be reduced

        ncp : int
            The new number of free collocation points for `var`. Must be
            less that the number of collocation points used in discretizing
            the model.

        contset : ``pyomo.dae.ContinuousSet``
            The :py:class:`ContinuousSet<pyomo.dae.ContinuousSet>` that was
            discretized and for which the `var` will have a reduced number
            of degrees of freedom

        