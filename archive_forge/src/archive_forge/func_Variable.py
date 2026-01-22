import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def Variable(shape, backend=None):
    """Create a ``LazyArray`` from a shape only, representing a leaf node
    in the computational graph. It can only act as a placeholder for data.
    """
    return LazyArray.from_shape(shape, backend=backend)