import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
@property
def _reference(self):
    if self._conv_ref is None:
        return self
    else:
        return self._conv_ref