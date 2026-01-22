import operator
import sys
from types import BuiltinFunctionType, BuiltinMethodType, FunctionType, MethodType
import numpy as np
import pandas as pd
import param
from ..core.data import PandasInterface
from ..core.dimension import Dimension
from ..core.util import flatten, resolve_dependent_value, unique_iterator
def _compute_data(self, data, drop_index, compute):
    if drop_index:
        data = data.data
    if hasattr(data, 'compute') and compute:
        data = data.compute()
    return data