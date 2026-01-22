import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
@staticmethod
def _from_subplot_args(figure, args):
    """
        Construct a `.SubplotSpec` from a parent `.Figure` and either

        - a `.SubplotSpec` -- returned as is;
        - one or three numbers -- a MATLAB-style subplot specifier.
        """
    if len(args) == 1:
        arg, = args
        if isinstance(arg, SubplotSpec):
            return arg
        elif not isinstance(arg, Integral):
            raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}')
        try:
            rows, cols, num = map(int, str(arg))
        except ValueError:
            raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}') from None
    elif len(args) == 3:
        rows, cols, num = args
    else:
        raise _api.nargs_error('subplot', takes='1 or 3', given=len(args))
    gs = GridSpec._check_gridspec_exists(figure, rows, cols)
    if gs is None:
        gs = GridSpec(rows, cols, figure=figure)
    if isinstance(num, tuple) and len(num) == 2:
        if not all((isinstance(n, Integral) for n in num)):
            raise ValueError(f'Subplot specifier tuple must contain integers, not {num}')
        i, j = num
    else:
        if not isinstance(num, Integral) or num < 1 or num > rows * cols:
            raise ValueError(f'num must be an integer with 1 <= num <= {rows * cols}, not {num!r}')
        i = j = num
    return gs[i - 1:j]