import numpy as np
from . import markup
from .quantity import Quantity, scale_other_units
from .registry import unit_registry
from .decorators import with_doc

        Return a tuple for pickling a Quantity.
        