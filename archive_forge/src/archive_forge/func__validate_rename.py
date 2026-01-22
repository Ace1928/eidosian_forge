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
def _validate_rename(self, mapping):
    pnames = [p.name for p in self.parameters]
    for k, v in mapping.items():
        n = k[1] if isinstance(k, tuple) else k
        if n not in pnames:
            raise KeyError(f'Cannot rename {n!r} as it is not a stream parameter')
        if n != v and v in pnames:
            raise KeyError('Cannot rename to %r as it clashes with a stream parameter of the same name' % v)
    return mapping