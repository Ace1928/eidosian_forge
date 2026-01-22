import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
class MyClass2:
    __array_priority__ = 100

    def __mul__(self, other):
        return 'Me2mul'

    def __rmul__(self, other):
        return 'Me2rmul'

    def __rdiv__(self, other):
        return 'Me2rdiv'
    __rtruediv__ = __rdiv__