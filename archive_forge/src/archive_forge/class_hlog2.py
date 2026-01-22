import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hlog2(Stub):
    """hlog2(a)

        Calculate logarithm base 2 in round to nearest even mode. Supported
        on fp16 operands only.

        Returns the logarithm base 2 result.

        """