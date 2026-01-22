from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class VLine(Annotation):
    """Vertical line annotation at the given position."""
    group = param.String(default='VLine', constant=True)
    x = param.ClassSelector(default=0, class_=(Number,) + datetime_types, doc='\n       The x-position of the VLine which make be numeric or a timestamp.')
    __pos_params = ['x']

    def __init__(self, x, **params):
        if isinstance(x, np.ndarray) and x.size == 1:
            x = np.atleast_1d(x)[0]
        super().__init__(x, x=x, **params)

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        index = self.get_dimension_index(dimension)
        if index == 0:
            return np.array([self.data])
        elif index == 1:
            return np.array([np.nan])
        else:
            return super().dimension_values(dimension)