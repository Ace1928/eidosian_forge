import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
def _notclose(fs, rtol=_rtol, atol=_xtol):
    notclosefvals = all(fs) and all(np.isfinite(fs)) and (not any((any(np.isclose(_f, fs[i + 1:], rtol=rtol, atol=atol)) for i, _f in enumerate(fs[:-1]))))
    return notclosefvals