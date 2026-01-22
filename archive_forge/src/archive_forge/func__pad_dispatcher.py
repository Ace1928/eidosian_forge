import numpy as np
from numpy.core.overrides import array_function_dispatch
from numpy.lib.index_tricks import ndindex
def _pad_dispatcher(array, pad_width, mode=None, **kwargs):
    return (array,)