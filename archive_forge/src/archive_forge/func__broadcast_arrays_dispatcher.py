import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
def _broadcast_arrays_dispatcher(*args, subok=None):
    return args