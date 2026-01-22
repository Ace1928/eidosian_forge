from numbers import Number
import numpy as np
import param
from ..core import Dimension, Element, Element2D
from ..core.data import Dataset
from ..core.util import datetime_types
class HLine(Annotation):
    """Horizontal line annotation at the given position."""
    group = param.String(default='HLine', constant=True)
    y = param.ClassSelector(default=0, class_=(Number,) + datetime_types, doc='\n       The y-position of the HLine which make be numeric or a timestamp.')
    __pos_params = ['y']

    def __init__(self, y, **params):
        if isinstance(y, np.ndarray) and y.size == 1:
            y = np.atleast_1d(y)[0]
        super().__init__(y, y=y, **params)

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
            return np.array([np.nan])
        elif index == 1:
            return np.array([self.data])
        else:
            return super().dimension_values(dimension)