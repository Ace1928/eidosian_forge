from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
def _dataslice(self, data, indices):
    """
        Returns slice of data element if the item is deep
        indexable. Warns if attempting to slice an object that has not
        been declared deep indexable.
        """
    if self._deep_indexable and isinstance(data, Dimensioned) and indices:
        return data[indices]
    elif len(indices) > 0:
        self.param.warning('Cannot index into data element, extra data indices ignored.')
    return data