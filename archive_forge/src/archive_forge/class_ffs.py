import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class ffs(Stub):
    """
    ffs(x)

    Returns the position of the first (least significant) bit set to 1 in x,
    where the least significant bit position is 1. ffs(0) returns 0.
    """