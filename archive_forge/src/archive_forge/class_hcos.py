import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hcos(Stub):
    """hsin(a)

        Calculate cosine in round to nearest even mode. Supported on fp16
        operands only.

        Returns the cosine result.

        """