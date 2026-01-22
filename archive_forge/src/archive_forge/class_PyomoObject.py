from pyomo.common.autoslots import AutoSlots
class PyomoObject(AutoSlots.Mixin):
    __slots__ = ()

    def is_component_type(self):
        """Return True if this class is a Pyomo component"""
        return False

    def is_numeric_type(self):
        """Return True if this class is a Pyomo numeric object"""
        return False

    def is_parameter_type(self):
        """Return False unless this class is a parameter object"""
        return False

    def is_variable_type(self):
        """Return False unless this class is a variable object"""
        return False

    def is_expression_type(self, expression_system=None):
        """Return True if this numeric value is an expression"""
        return False

    def is_named_expression_type(self):
        """Return True if this numeric value is a named expression"""
        return False

    def is_logical_type(self):
        """Return True if this class is a Pyomo Boolean object.

        Boolean objects include constants, variables, or logical expressions.
        """
        return False

    def is_reference(self):
        """Return True if this object is a reference."""
        return False