from distutils.version import StrictVersion
import functools
import ast
import numpy as np
from . import operator
from . import numpy as _mx_np  # pylint: disable=reimported
from .util import np_array, use_np
from .numpy.utils import _STR_2_DTYPE_
from .ndarray.numpy import _internal as _nd_npi
from .symbol.numpy import _internal as _sym_npi
def _save_op(mod):
    if hasattr(mod, op_name):
        raise ValueError('Duplicate name {} found in module {}'.format(op_name, str(mod)))
    op = functools.partial(mod.Custom, op_type=op_name)
    setattr(mod, op_name, op)