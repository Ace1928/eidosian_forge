from numpy.core import integer, empty, arange, asarray, roll
from numpy.core.overrides import array_function_dispatch, set_module
def _fftshift_dispatcher(x, axes=None):
    return (x,)