from decimal import Decimal
from numbers import Number
import numpy as np
from numpy import ma
from matplotlib import cbook
class ConversionInterface:
    """
    The minimal interface for a converter to take custom data types (or
    sequences) and convert them to values Matplotlib can use.
    """

    @staticmethod
    def axisinfo(unit, axis):
        """Return an `.AxisInfo` for the axis with the specified units."""
        return None

    @staticmethod
    def default_units(x, axis):
        """Return the default unit for *x* or ``None`` for the given axis."""
        return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        Convert *obj* using *unit* for the specified *axis*.

        If *obj* is a sequence, return the converted sequence.  The output must
        be a sequence of scalars that can be used by the numpy array layer.
        """
        return obj