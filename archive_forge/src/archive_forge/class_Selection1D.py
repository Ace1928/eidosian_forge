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
class Selection1D(LinkedStream):
    """
    A stream representing a 1D selection of objects by their index.
    """
    index = param.List(default=[], allow_None=True, constant=True, doc='\n        Indices into a 1D datastructure.')