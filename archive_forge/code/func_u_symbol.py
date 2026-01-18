import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
@property
def u_symbol(self):
    if self._u_symbol:
        return self._u_symbol
    else:
        return self.symbol