import functools
import numpy.core.numeric as _nx
from numpy.core.numeric import asarray, zeros, array, asanyarray
from numpy.core.fromnumeric import reshape, transpose
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
from numpy.core import vstack, atleast_3d
from numpy.core.numeric import normalize_axis_tuple
from numpy.core.shape_base import _arrays_for_stack_dispatcher
from numpy.lib.index_tricks import ndindex
from numpy.matrixlib.defmatrix import matrix  # this raises all the right alarm bells
def _column_stack_dispatcher(tup):
    return _arrays_for_stack_dispatcher(tup)