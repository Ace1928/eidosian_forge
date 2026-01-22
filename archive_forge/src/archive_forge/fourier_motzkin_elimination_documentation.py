from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
Function that solves a sequence of LPs problems to check if
        constraints are implied by each other. Deletes any that are.

        Parameters
        ----------------
        m: A model, already transformed with FME. Note that if constraints
           have been added, activated, or deactivated, we will check for
           redundancy against the whole active part of the model. If you call
           this straight after FME, you are only checking within the projected
           constraints, but otherwise it is up to the user.
        solver_factory: A SolverFactory object (constructed with a solver
                        which can solve the continuous relaxation of the
                        active constraints on the model. That is, if you
                        had nonlinear constraints unrelated to the variables
                        being projected, you need to either deactivate them or
                        provide a solver which will do the right thing.)
        projected_constraints: The ConstraintList of projected constraints.
                               Default is None, in which case we assume that
                               the FME transformation was called without
                               specifying their name, so will look for them on
                               the private transformation block.
        tolerance: Tolerance at which we decide a constraint is implied by the
                   others. Default is 0, meaning we remove the constraint if
                   the LP solve finds the constraint can be tight but not
                   violated. Setting this to a small positive value would
                   remove constraints more conservatively. Setting it to a
                   negative value would result in a relaxed problem.
        