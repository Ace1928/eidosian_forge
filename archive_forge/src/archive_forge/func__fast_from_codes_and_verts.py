import copy
from functools import lru_cache
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from . import _api, _path
from .cbook import _to_unmasked_float_array, simple_linear_interpolation
from .bezier import BezierSegment
@classmethod
def _fast_from_codes_and_verts(cls, verts, codes, internals_from=None):
    """
        Create a Path instance without the expense of calling the constructor.

        Parameters
        ----------
        verts : array-like
        codes : array
        internals_from : Path or None
            If not None, another `Path` from which the attributes
            ``should_simplify``, ``simplify_threshold``, and
            ``interpolation_steps`` will be copied.  Note that ``readonly`` is
            never copied, and always set to ``False`` by this constructor.
        """
    pth = cls.__new__(cls)
    pth._vertices = _to_unmasked_float_array(verts)
    pth._codes = codes
    pth._readonly = False
    if internals_from is not None:
        pth._should_simplify = internals_from._should_simplify
        pth._simplify_threshold = internals_from._simplify_threshold
        pth._interpolation_steps = internals_from._interpolation_steps
    else:
        pth._should_simplify = True
        pth._simplify_threshold = mpl.rcParams['path.simplify_threshold']
        pth._interpolation_steps = 1
    return pth