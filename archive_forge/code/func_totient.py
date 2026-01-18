import math
import operator
from collections import deque
from collections.abc import Sized
from functools import partial, reduce
from itertools import (
from random import randrange, sample, choice
from sys import hexversion
def totient(n):
    """Return the count of natural numbers up to *n* that are coprime with *n*.

    >>> totient(9)
    6
    >>> totient(12)
    4
    """
    for p in unique_justseen(factor(n)):
        n = n // p * (p - 1)
    return n