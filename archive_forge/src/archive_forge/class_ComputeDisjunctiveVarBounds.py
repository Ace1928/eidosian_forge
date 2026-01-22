from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt, BoundsManager
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.core.expr import identify_variables
from pyomo.core import Constraint, Objective, TransformationFactory, minimize, value
from pyomo.opt import SolverFactory
from pyomo.gdp.disjunct import Disjunct
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.opt import TerminationCondition as tc
@TransformationFactory.register('contrib.compute_disj_var_bounds', doc='Compute disjunctive bounds in a given model.')
class ComputeDisjunctiveVarBounds(Transformation):
    """Compute disjunctive bounds in a given model.

    Tries to compute the disjunctive bounds for all variables found in
    constraints that are in disjuncts under the given model.

    Two strategies are available to compute the disjunctive bounds:
     - Feasibility-based bounds tightening using the contrib.fbbt package. (Default)
     - Optimality-based bounds tightening by solving the linear relaxation of the model.

    This transformation introduces ComponentMap objects named _disj_var_bounds to
    each Disjunct and the top-level model object. These map var --> (var.disj_lb, var.disj_ub)
    for each disjunctive scope.

    Args:
        model (Component): The model under which to look for disjuncts.
        solver (string): The solver to use for OBBT, or None for FBBT.

    """

    def _apply_to(self, model, solver=None):
        """Apply the transformation.

        Args:
            model: Pyomo model object on which to compute disjuctive bounds.

        """
        if solver is not None:
            disjunctive_obbt(model, solver)
        else:
            disjunctive_fbbt(model)