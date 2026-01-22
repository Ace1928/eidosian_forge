import sys
import types
import logging
from weakref import ref as weakref_ref
from pyomo.common.pyomo_typing import overload
from pyomo.common.autoslots import AutoSlots
from pyomo.common.deprecation import deprecation_warning, RenamedClass
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.numeric_types import native_types, value as expr_value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import NumericValue
from pyomo.core.base.component import ComponentData, ModelComponentFactory
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.indexed_component import (
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.misc import apply_indexed_rule, apply_parameterized_indexed_rule
from pyomo.core.base.set import Reals, _AnySet, SetInitializer
from pyomo.core.base.units_container import units
from pyomo.core.expr import GetItemExpression
class _ParamData(ComponentData, NumericValue):
    """
    This class defines the data for a mutable parameter.

    Constructor Arguments:
        owner       The Param object that owns this data.
        value       The value of this parameter.

    Public Class Attributes:
        value       The numeric value of this variable.
    """
    __slots__ = ('_value',)

    def __init__(self, component):
        self._component = weakref_ref(component)
        self._index = NOTSET
        self._value = Param.NoValue

    def clear(self):
        """Clear the data in this component"""
        self._value = Param.NoValue

    def set_value(self, value, idx=NOTSET):
        _comp = self.parent_component()
        if value.__class__ in native_types:
            pass
        elif _comp._units is not None:
            _src_magnitude = expr_value(value)
            if value.__class__ in native_types:
                value = _src_magnitude
            else:
                _src_units = units.get_units(value)
                value = units.convert_value(num_value=_src_magnitude, from_units=_src_units, to_units=_comp._units)
        old_value, self._value = (self._value, value)
        try:
            _comp._validate_value(idx, value, data=self)
        except:
            self._value = old_value
            raise

    def __call__(self, exception=True):
        """
        Return the value of this object.
        """
        if self._value is Param.NoValue:
            if exception:
                raise ValueError('Error evaluating Param value (%s):\n\tThe Param value is currently set to an invalid value.  This is\n\ttypically from a scalar Param or mutable Indexed Param without\n\tan initial or default value.' % (self.name,))
            else:
                return None
        return self._value

    @property
    def value(self):
        """Return the value for this variable."""
        return self()

    @value.setter
    def value(self, val):
        """Set the value for this variable."""
        self.set_value(val)

    def get_units(self):
        """Return the units for this ParamData"""
        return self.parent_component()._units

    def is_fixed(self):
        """
        Returns True because this value is fixed.
        """
        return True

    def is_constant(self):
        """
        Returns False because this is not a constant in an expression.
        """
        return False

    def is_parameter_type(self):
        """
        Returns True because this is a parameter object.
        """
        return True

    def _compute_polynomial_degree(self, result):
        """
        Returns 0 because this object can never reference variables.
        """
        return 0