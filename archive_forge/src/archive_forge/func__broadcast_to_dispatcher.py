import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
def _broadcast_to_dispatcher(array, shape, subok=None):
    return (array,)