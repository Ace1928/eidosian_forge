import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
class UnequalIterablesError(ValueError):

    def __init__(self, details=None):
        msg = 'Iterables have different lengths'
        if details is not None:
            msg += ': index 0 has length {}; index {} has length {}'.format(*details)
        super().__init__(msg)