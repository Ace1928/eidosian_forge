from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def _update_props(self, props, errfmt):
    """
        Helper for `.Artist.set` and `.Artist.update`.

        *errfmt* is used to generate error messages for invalid property
        names; it gets formatted with ``type(self)`` and the property name.
        """
    ret = []
    with cbook._setattr_cm(self, eventson=False):
        for k, v in props.items():
            if k == 'axes':
                ret.append(setattr(self, k, v))
            else:
                func = getattr(self, f'set_{k}', None)
                if not callable(func):
                    raise AttributeError(errfmt.format(cls=type(self), prop_name=k))
                ret.append(func(v))
    if ret:
        self.pchanged()
        self.stale = True
    return ret