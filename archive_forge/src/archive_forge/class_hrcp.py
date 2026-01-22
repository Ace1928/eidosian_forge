import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hrcp(Stub):
    """hrcp(a)

        Calculate the reciprocal of the input argument in round to nearest
        even mode. Supported on fp16 operands only.

        Returns the reciprocal result.

        """