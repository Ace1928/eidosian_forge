import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _add_reduced_axes(res, reduced_axes, keepdims):
    """
    Add reduced axes back to all the arrays in the result object
    if keepdims = True.
    """
    return [np.expand_dims(output, reduced_axes) for output in res] if keepdims else res