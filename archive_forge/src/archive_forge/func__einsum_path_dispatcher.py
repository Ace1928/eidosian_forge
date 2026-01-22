import itertools
import operator
from numpy.core.multiarray import c_einsum
from numpy.core.numeric import asanyarray, tensordot
from numpy.core.overrides import array_function_dispatch
def _einsum_path_dispatcher(*operands, optimize=None, einsum_call=None):
    return operands