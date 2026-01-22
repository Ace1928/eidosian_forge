import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.overrides import array_function_dispatch, set_module
def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        new_array = new_array.view(type=type(original_array))
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array