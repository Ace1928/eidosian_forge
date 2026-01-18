from __future__ import division  # Many analytical derivatives depend on this
from builtins import map
import math
import sys
import itertools
import uncertainties.core as uncert_core
from uncertainties.core import (to_affine_scalar, AffineScalarFunc,
def wrapped_fsum():
    """
    Return an uncertainty-aware version of math.fsum, which must
    be contained in _original_func.
    """
    flat_fsum = lambda *args: original_func(args)
    flat_fsum_wrap = uncert_core.wrap(flat_fsum, itertools.repeat(lambda *args: 1))
    return wraps(lambda arg_list: flat_fsum_wrap(*arg_list), original_func)