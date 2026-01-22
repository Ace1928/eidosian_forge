import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_unit_for_equivalent_children(self, node, child_units):
    """
        Return (and test) the units corresponding to an expression node in the
        expression tree where all children should have the same units (e.g. sum).

        Parameters
        ----------
        node : Pyomo expression node
            The parent node of the children

        child_units : list
           This is a list of pint units (one for each of the children)

        Returns
        -------
        : pint unit
        """
    assert bool(child_units)
    pint_unit_0 = child_units[0]
    for pint_unit_i in child_units:
        if not self._equivalent_pint_units(pint_unit_0, pint_unit_i):
            raise InconsistentUnitsError(pint_unit_0, pint_unit_i, 'Error in units found in expression: %s' % (node,))
    return pint_unit_0