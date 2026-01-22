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
class SelectionXY(BoundsXY):
    """
    A stream representing the selection along the x-axis and y-axis.
    Unlike a BoundsXY stream, this stream returns range or categorical
    selections.
    """
    bounds = param.Tuple(default=None, constant=True, length=4, allow_None=True, doc='\n        Bounds defined as (left, bottom, right, top) tuple.')
    x_selection = param.ClassSelector(class_=(tuple, list), allow_None=True, constant=True, doc='\n      The current selection along the x-axis, either a numerical range\n      defined as a tuple or a list of categories.')
    y_selection = param.ClassSelector(class_=(tuple, list), allow_None=True, constant=True, doc='\n      The current selection along the y-axis, either a numerical range\n      defined as a tuple or a list of categories.')