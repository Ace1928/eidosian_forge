import math
import operator
import warnings
from collections import deque
from collections.abc import Sized
from functools import reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
Yield the prime factors of n.
    >>> list(factor(360))
    [2, 2, 2, 3, 3, 5]
    