import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
class Dimensionless(UnitQuantity):
    _primary_order = 100

    def __init__(self, name, definition=None):
        self._name = name
        if definition is None:
            definition = self
        self._definition = definition
        self._format_order = (self._primary_order, self._secondary_order)
        self.__class__._secondary_order += 1
        unit_registry[name] = self

    def __reduce__(self):
        """
        Return a tuple for pickling a UnitQuantity.
        """
        return (type(self), (self._name,), self.__getstate__())

    @property
    def _dimensionality(self):
        return Dimensionality()