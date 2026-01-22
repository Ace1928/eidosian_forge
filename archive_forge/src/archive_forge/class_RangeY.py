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
class RangeY(LinkedStream):
    """
    Axis range along y-axis in data coordinates.
    """
    y_range = param.Tuple(default=None, length=2, constant=True, doc='\n      Range of the y-axis of a plot in data coordinates')