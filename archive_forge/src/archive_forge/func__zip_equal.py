import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def _zip_equal(*iterables):
    try:
        first_size = len(iterables[0])
        for i, it in enumerate(iterables[1:], 1):
            size = len(it)
            if size != first_size:
                break
        else:
            return zip(*iterables)
        raise UnequalIterablesError(details=(first_size, i, size))
    except TypeError:
        return _zip_equal_generator(iterables)