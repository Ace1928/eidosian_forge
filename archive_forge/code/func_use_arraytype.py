from pygame.pixelcopy import (
import numpy
from numpy import (
import warnings  # will be removed in the future
def use_arraytype(arraytype):
    """pygame.surfarray.use_arraytype(arraytype): return None

    DEPRECATED - only numpy arrays are now supported.
    """
    warnings.warn(DeprecationWarning('only numpy arrays are now supported, this function will be removed in a future version of the module'))
    arraytype = arraytype.lower()
    if arraytype != 'numpy':
        raise ValueError('invalid array type')