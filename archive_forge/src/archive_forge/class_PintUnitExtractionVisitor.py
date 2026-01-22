import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class PintUnitExtractionVisitor(EXPR.StreamBasedExpressionVisitor):

    def __init__(self, pyomo_units_container, units_equivalence_tolerance=1e-12):
        """
        Visitor class used to determine units of an expression. Do not use
        this class directly, but rather use
        "py:meth:`PyomoUnitsContainer.assert_units_consistent`
        or :py:meth:`PyomoUnitsContainer.get_units`

        Parameters
        ----------
        pyomo_units_container : PyomoUnitsContainer
           Instance of the PyomoUnitsContainer that was used for the units
           in the expressions. Pyomo does not support "mixing" units from
           different containers

        units_equivalence_tolerance : float (default 1e-12)
            Floating point tolerance used when deciding if units are equivalent
            or not.

        Notes
        -----
        This class inherits from the :class:`StreamBasedExpressionVisitor` to implement
        a walker that returns the pyomo units and pint units corresponding to an
        expression.

        There are class attributes (dicts) that map the expression node type to the
        particular method that should be called to return the units of the node based
        on the units of its child arguments. This map is used in exitNode.
        """
        super(PintUnitExtractionVisitor, self).__init__()
        self._pyomo_units_container = pyomo_units_container
        self._pint_dimensionless = None
        self._equivalent_pint_units = pyomo_units_container._equivalent_pint_units
        self._equivalent_to_dimensionless = pyomo_units_container._equivalent_to_dimensionless

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

    def _get_unit_for_product(self, node, child_units):
        """
        Return (and test) the units corresponding to a product expression node
        in the expression tree.

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
        assert len(child_units) == 2
        pint_unit = child_units[0] * child_units[1]
        if hasattr(pint_unit, 'units'):
            return pint_unit.units
        return pint_unit

    def _get_unit_for_division(self, node, child_units):
        """
        Return (and test) the units corresponding to a division expression node
        in the expression tree.

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
        assert len(child_units) == 2
        return child_units[0] / child_units[1]

    def _get_unit_for_pow(self, node, child_units):
        """
        Return (and test) the units corresponding to a pow expression node
        in the expression tree.

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
        assert len(child_units) == 2
        if not self._equivalent_to_dimensionless(child_units[1]):
            raise UnitsError(f'Error in sub-expression: {node}. Exponents in a pow expression must be dimensionless.')
        exponent = node.args[1]
        if type(exponent) in nonpyomo_leaf_types:
            return child_units[0] ** value(exponent)
        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pint_dimensionless
        if not exponent.is_fixed():
            raise UnitsError(f'The base of an exponent has units {child_units[0]}, but the exponent is not a fixed numerical value.')
        return child_units[0] ** value(exponent)

    def _get_unit_for_single_child(self, node, child_units):
        """
        Return (and test) the units corresponding to a unary operation (e.g. negation)
        expression node in the expression tree.

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
        assert len(child_units) == 1
        return child_units[0]

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

    def _get_dimensionless_with_dimensionless_children(self, node, child_units):
        """
        Check to make sure that any child arguments are unitless /
        dimensionless (for functions like exp()) and return dimensionless.

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
        for pint_unit in child_units:
            if not self._equivalent_to_dimensionless(pint_unit):
                raise UnitsError(f'Expected no units or dimensionless units in {node}, but found {pint_unit}.')
        return self._pint_dimensionless

    def _get_dimensionless_no_children(self, node, child_units):
        """
        Check to make sure the length of child_units is zero, and returns
        dimensionless. Used for leaf nodes that should not have any units.

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
        assert len(child_units) == 0
        assert type(node) is IndexTemplate
        return self._pint_dimensionless

    def _get_unit_for_unary_function(self, node, child_units):
        """
        Return (and test) the units corresponding to a unary function expression node
        in the expression tree. Checks that child_units is of length 1
        and calls the appropriate method from the unary function method map.

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
        assert len(child_units) == 1
        func_name = node.getname()
        node_func = self.unary_function_method_map.get(func_name, None)
        if node_func is None:
            raise TypeError(f'An unhandled unary function: {func_name} was encountered while retrieving the units of expression {node}')
        return node_func(self, node, child_units)

    def _get_unit_for_expr_if(self, node, child_units):
        """
        Return (and test) the units corresponding to an Expr_if expression node
        in the expression tree. The _if relational expression is validated and
        the _then/_else are checked to ensure they have the same units. Also checks
        to make sure length of child_units is 3

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
        assert len(child_units) == 3
        if not self._equivalent_pint_units(child_units[1], child_units[2]):
            raise InconsistentUnitsError(child_units[1], child_units[2], 'Error in units found in expression: %s' % (node,))
        return child_units[1]

    def _get_dimensionless_with_radians_child(self, node, child_units):
        """
        Return (and test) the units corresponding to a trig function expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the units of that child expression are unitless or radians and
        returns dimensionless for the units.

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
        assert len(child_units) == 1
        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pint_dimensionless
        if self._equivalent_pint_units(child_units[0], self._pyomo_units_container._pint_registry.radian):
            return self._pint_dimensionless
        raise UnitsError('Expected radians or dimensionless in argument to function in expression %s, but found %s' % (node, child_units[0]))

    def _get_radians_with_dimensionless_child(self, node, child_units):
        """
        Return (and test) the units corresponding to an inverse trig expression node
        in the expression tree. Checks that the length of child_units is 1
        and that the child argument is dimensionless, and returns radians

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
        assert len(child_units) == 1
        if self._equivalent_to_dimensionless(child_units[0]):
            return self._pyomo_units_container._pint_registry.radian
        raise UnitsError(f'Expected dimensionless argument to function in expression {node}, but found {child_units[0]}')

    def _get_unit_sqrt(self, node, child_units):
        """
        Return (and test) the units corresponding to a sqrt expression node
        in the expression tree. Checks that the length of child_units is one.

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
        assert len(child_units) == 1
        return child_units[0] ** 0.5
    node_type_method_map = {EXPR.EqualityExpression: _get_unit_for_equivalent_children, EXPR.InequalityExpression: _get_unit_for_equivalent_children, EXPR.RangedExpression: _get_unit_for_equivalent_children, EXPR.SumExpression: _get_unit_for_equivalent_children, EXPR.NPV_SumExpression: _get_unit_for_equivalent_children, EXPR.ProductExpression: _get_unit_for_product, EXPR.MonomialTermExpression: _get_unit_for_product, EXPR.NPV_ProductExpression: _get_unit_for_product, EXPR.DivisionExpression: _get_unit_for_division, EXPR.NPV_DivisionExpression: _get_unit_for_division, EXPR.PowExpression: _get_unit_for_pow, EXPR.NPV_PowExpression: _get_unit_for_pow, EXPR.NegationExpression: _get_unit_for_single_child, EXPR.NPV_NegationExpression: _get_unit_for_single_child, EXPR.AbsExpression: _get_unit_for_single_child, EXPR.NPV_AbsExpression: _get_unit_for_single_child, EXPR.UnaryFunctionExpression: _get_unit_for_unary_function, EXPR.NPV_UnaryFunctionExpression: _get_unit_for_unary_function, EXPR.Expr_ifExpression: _get_unit_for_expr_if, IndexTemplate: _get_dimensionless_no_children, EXPR.Numeric_GetItemExpression: _get_dimensionless_with_dimensionless_children, EXPR.NPV_Numeric_GetItemExpression: _get_dimensionless_with_dimensionless_children, EXPR.ExternalFunctionExpression: _get_units_ExternalFunction, EXPR.NPV_ExternalFunctionExpression: _get_units_ExternalFunction, EXPR.LinearExpression: _get_unit_for_equivalent_children}
    unary_function_method_map = {'log': _get_dimensionless_with_dimensionless_children, 'log10': _get_dimensionless_with_dimensionless_children, 'sin': _get_dimensionless_with_radians_child, 'cos': _get_dimensionless_with_radians_child, 'tan': _get_dimensionless_with_radians_child, 'sinh': _get_dimensionless_with_radians_child, 'cosh': _get_dimensionless_with_radians_child, 'tanh': _get_dimensionless_with_radians_child, 'asin': _get_radians_with_dimensionless_child, 'acos': _get_radians_with_dimensionless_child, 'atan': _get_radians_with_dimensionless_child, 'exp': _get_dimensionless_with_dimensionless_children, 'sqrt': _get_unit_sqrt, 'asinh': _get_radians_with_dimensionless_child, 'acosh': _get_radians_with_dimensionless_child, 'atanh': _get_radians_with_dimensionless_child, 'ceil': _get_unit_for_single_child, 'floor': _get_unit_for_single_child}

    def initializeWalker(self, expr):
        self._pint_dimensionless = self._pyomo_units_container._pint_dimensionless
        walk, result = self.beforeChild(None, expr, 0)
        if not walk:
            result = self.finalizeResult(result)
        return (walk, result)

    def beforeChild(self, node, child, child_idx):
        ctype = child.__class__
        if ctype in native_types or ctype in pyomo_constant_types:
            return (False, self._pint_dimensionless)
        if child.is_expression_type():
            return (True, None)
        if ctype is _PyomoUnit:
            return (False, child._get_pint_unit())
        elif hasattr(child, 'get_units'):
            pyomo_unit = child.get_units()
            pint_unit = self._pyomo_units_container._get_pint_units(pyomo_unit)
            return (False, pint_unit)
        return (True, None)

    def exitNode(self, node, data):
        """Visitor callback when moving up the expression tree.

        Callback for
        :class:`pyomo.core.current.StreamBasedExpressionVisitor`. This
        method is called when moving back up the tree in a depth first
        search.

        """
        node_func = self.node_type_method_map.get(node.__class__, None)
        if node_func is not None:
            return node_func(self, node, data)
        if hasattr(node, 'is_named_expression_type') and node.is_named_expression_type():
            pint_unit = self._get_unit_for_single_child(node, data)
            return pint_unit
        raise TypeError(f'An unhandled expression node type: {type(node)} was encountered while retrieving the units of expression {node}')

    def finalizeResult(self, result):
        if hasattr(result, 'units'):
            return result.units
        return result