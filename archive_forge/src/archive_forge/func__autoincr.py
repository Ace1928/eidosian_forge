from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
def _autoincr():
    n = next(_typecodes)
    assert n < 2 ** 32, 'Limited to 4 billion types'
    return n