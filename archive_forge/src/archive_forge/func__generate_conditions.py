from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _generate_conditions(self, map_slice):
    """
        Generates filter conditions used for slicing the data structure.
        """
    conditions = []
    for dim, dim_slice in zip(self.kdims, map_slice):
        if isinstance(dim_slice, slice):
            start, stop = (dim_slice.start, dim_slice.stop)
            if dim.values:
                values = dim.values
                dim_slice = slice(None if start is None else values.index(start), None if stop is None else values.index(stop))
            if dim_slice == slice(None):
                conditions.append(self._all_condition())
            elif start is None:
                conditions.append(self._upto_condition(dim_slice))
            elif stop is None:
                conditions.append(self._from_condition(dim_slice))
            else:
                conditions.append(self._range_condition(dim_slice))
        elif isinstance(dim_slice, (set, list)):
            if dim.values:
                dim_slice = [dim.values.index(dim_val) for dim_val in dim_slice]
            conditions.append(self._values_condition(dim_slice))
        elif dim_slice is Ellipsis:
            conditions.append(self._all_condition())
        elif callable(dim_slice):
            conditions.append(dim_slice)
        elif isinstance(dim_slice, tuple):
            raise IndexError('Keys may only be selected with sets or lists, not tuples.')
        else:
            if dim.values:
                dim_slice = dim.values.index(dim_slice)
            conditions.append(self._value_condition(dim_slice))
    return conditions