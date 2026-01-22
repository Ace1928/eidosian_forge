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
@contextlib.contextmanager
def _cm_set(self, **kwargs):
    """
        `.Artist.set` context-manager that restores original values at exit.
        """
    orig_vals = {k: getattr(self, f'get_{k}')() for k in kwargs}
    try:
        self.set(**kwargs)
        yield
    finally:
        self.set(**orig_vals)