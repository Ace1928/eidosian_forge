import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
class BoundsY(LinkedStream):
    """
    A stream representing the bounds of a box selection as an
    tuple of the bottom and top coordinates.
    """
    boundsy = param.Tuple(default=None, constant=True, length=2, allow_None=True, doc='\n        Bounds defined as (bottom, top) tuple.')