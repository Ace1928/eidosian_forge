import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
def _get_units_ExternalFunction(self, node, child_units):
    """
        Check to make sure that any child arguments are consistent with
        arg_units return the value from node.get_units() This
        was written for ExternalFunctionExpression where the external
        function has units assigned to its return value and arguments

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
    arg_units = node.get_arg_units()
    dless = self._pint_dimensionless
    if arg_units is None:
        arg_units = [dless] * len(child_units)
    else:
        arg_units = list(arg_units)
        for i, a in enumerate(arg_units):
            arg_units[i] = self._pyomo_units_container._get_pint_units(a)
    for arg_unit, pint_unit in zip(arg_units, child_units):
        assert arg_unit is not None
        if not self._equivalent_pint_units(arg_unit, pint_unit):
            raise InconsistentUnitsError(arg_unit, pint_unit, 'Inconsistent units found in ExternalFunction.')
    return self._pyomo_units_container._get_pint_units(node.get_units())