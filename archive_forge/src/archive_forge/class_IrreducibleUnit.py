import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
class IrreducibleUnit(UnitQuantity):
    _default_unit = None

    def __init__(self, name, definition=None, symbol=None, u_symbol=None, aliases=[], doc=None):
        super().__init__(name, definition, symbol, u_symbol, aliases, doc)
        cls = type(self)
        if cls._default_unit is None:
            cls._default_unit = self

    @property
    def simplified(self):
        return self.view(Quantity).rescale(self.get_default_unit())

    @classmethod
    def get_default_unit(cls):
        return cls._default_unit

    @classmethod
    def set_default_unit(cls, unit):
        if unit is None:
            return
        if isinstance(unit, str):
            unit = unit_registry[unit]
        try:
            get_conversion_factor(cls._default_unit, unit)
        except ValueError:
            raise TypeError('default unit must be of same type')
        cls._default_unit = unit