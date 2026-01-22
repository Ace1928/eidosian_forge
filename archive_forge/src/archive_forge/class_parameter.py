from pyomo.core.expr.numvalue import is_numeric_data, NumericValue
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.container_utils import define_simple_containers
class parameter(IParameter):
    """A object for storing a mutable, numeric value that
    can be used to build a symbolic expression."""
    _ctype = IParameter
    __slots__ = ('_parent', '_storage_key', '_active', '_value', '__weakref__')

    def __init__(self, value=None):
        self._parent = None
        self._storage_key = None
        self._active = True
        self._value = value

    def __call__(self, exception=True):
        """Computes the numeric value of this object."""
        return self.value

    @property
    def value(self):
        """The value of the parameter"""
        return self._value

    @value.setter
    def value(self, value):
        self._value = value