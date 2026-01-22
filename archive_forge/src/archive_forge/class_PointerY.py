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
class PointerY(LinkedStream):
    """
    A pointer position along the y-axis in data coordinates which may be
    a numeric or categorical dimension.

    With the appropriate plotting backend, this corresponds to the
    position of the mouse/trackpad pointer. If the pointer is outside
    the plot bounds, the position is set to None.
    """
    y = param.ClassSelector(class_=pointer_types, default=None, constant=True, doc='\n           Pointer position along the y-axis in data coordinates')