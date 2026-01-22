import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class DataFrameInitializer(InitializerBase):
    """Initializer for pandas DataFrame values"""
    __slots__ = ('_df', '_column')

    def __init__(self, dataframe, column=None):
        self._df = dataframe
        if column is not None:
            self._column = column
        elif len(dataframe.columns) == 1:
            self._column = dataframe.columns[0]
        else:
            raise ValueError('Cannot construct DataFrameInitializer for DataFrame with multiple columns without also specifying the data column')

    def __call__(self, parent, idx):
        return self._df.at[idx, self._column]

    def contains_indices(self):
        return True

    def indices(self):
        return self._df.index