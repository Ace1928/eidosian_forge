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
class SelectMode(LinkedStream):
    mode = param.ObjectSelector(default='replace', constant=True, objects=['replace', 'append', 'intersect', 'subtract'], doc='\n        Defines what should happen when a new selection is made. The\n        default is to replace the existing selection. Other options\n        are to append to theselection, intersect with it or subtract\n        from it.')