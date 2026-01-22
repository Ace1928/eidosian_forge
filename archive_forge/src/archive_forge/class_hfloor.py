import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hfloor(Stub):
    """hfloor(a)

        Calculate the floor, the largest integer less than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the floor result.

        """