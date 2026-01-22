import logging
import sys
from pyomo.common.dependencies import attempt_import
from pyomo.common.modeling import NOTSET
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.visitor import ExpressionValueVisitor
import pyomo.core.expr as EXPR
class _PyomoUnit(NumericValue):
    """An object that represents a single unit in Pyomo (e.g., kg, meter)

    Users should not create instances of _PyomoUnit directly, but rather access
    units as attributes on an instance of a :class:`PyomoUnitsContainer`.
    This module contains a global PyomoUnitsContainer object :py:data:`units`.
    See module documentation for more information.
    """
    __slots__ = ('_pint_unit', '_pint_registry')
    __autoslot_mappers__ = {'_pint_unit': _pint_unit_mapper, '_pint_registry': _pint_registry_mapper}

    def __init__(self, pint_unit, pint_registry):
        super(_PyomoUnit, self).__init__()
        assert pint_unit is not None
        assert pint_registry is not None
        self._pint_unit = pint_unit
        self._pint_registry = pint_registry

    def _get_pint_unit(self):
        """Return the pint unit corresponding to this Pyomo unit."""
        return self._pint_unit

    def _get_pint_registry(self):
        """Return the pint registry (pint.UnitRegistry) object used to create this unit."""
        return self._pint_registry

    def getname(self, fully_qualified=False, name_buffer=None):
        """
        Returns the name of this unit as a string.
        Overloaded from: :py:class:`NumericValue`. See this class for a description of the
        arguments. The value of these arguments are ignored here.

        Returns
        -------
        : str
           Returns the name of the unit
        """
        return str(self)

    def is_constant(self):
        """
        Indicates if the NumericValue is constant and can be replaced with a plain old number
        Overloaded from: :py:class:`NumericValue`

        This method indicates if the NumericValue is a constant and can be replaced with a plain
        old number. Although units are, in fact, constant, we do NOT want this replaced - therefore
        we return False here to prevent replacement.

        Returns
        =======
        : bool
           False (This method always returns False)
        """
        return False

    def is_fixed(self):
        """
        Indicates if the NumericValue is fixed with respect to a "solver".
        Overloaded from: :py:class:`NumericValue`

        Indicates if the Unit should be treated as fixed. Since the Unit is always treated as
        a constant value of 1.0, it is fixed.

        Returns
        =======
        : bool
           True (This method always returns True)

        """
        return True

    def is_parameter_type(self):
        """This is not a parameter type (overloaded from NumericValue)"""
        return False

    def is_variable_type(self):
        """This is not a variable type (overloaded from NumericValue)"""
        return False

    def is_potentially_variable(self):
        """
        This is not potentially variable (does not and cannot contain a variable).
        Overloaded from NumericValue
        """
        return False

    def is_named_expression_type(self):
        """This is not a named expression (overloaded from NumericValue)"""
        return False

    def is_expression_type(self, expression_system=None):
        """This is a leaf, not an expression (overloaded from NumericValue)"""
        return False

    def is_component_type(self):
        """This is not a component type (overloaded from NumericValue)"""
        return False

    def is_indexed(self):
        """This is not indexed (overloaded from NumericValue)"""
        return False

    def _compute_polynomial_degree(self, result):
        """Returns the polynomial degree - since units are constants, they have degree of zero.
        Note that :py:meth:`NumericValue.polynomial_degree` calls this method.
        """
        return 0

    def __deepcopy__(self, memo):
        return self

    def __eq__(self, other):
        if other.__class__ is _PyomoUnit:
            return self._pint_registry is other._pint_registry and self._pint_unit == other._pint_unit
        return super().__eq__(other)

    def __str__(self):
        """Returns a string representing the unit"""
        retstr = u'{:~C}'.format(self._pint_unit)
        if retstr == '':
            retstr = 'dimensionless'
        return retstr

    def to_string(self, verbose=None, labeler=None, smap=None, compute_values=False):
        """
        Return a string representation of the expression tree.

        See documentation on :py:class:`NumericValue`

        Returns
        -------
        : bool
           A string representation for the expression tree.
        """
        _str = str(self)
        if any(map(_str.__contains__, ' */')):
            return '(' + _str + ')'
        else:
            return _str

    def __call__(self, exception=True):
        """Unit is treated as a constant value, and this method always returns 1.0

        Returns
        -------
        : float
           Returns 1.0
        """
        return 1.0

    @property
    def value(self):
        return 1.0

    def pprint(self, ostream=None, verbose=False):
        """Display a user readable string description of this object."""
        if ostream is None:
            ostream = sys.stdout
        ostream.write(str(self))