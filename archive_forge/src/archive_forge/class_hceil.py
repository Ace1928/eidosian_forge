import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hceil(Stub):
    """hceil(a)

        Calculate the ceil, the smallest integer greater than or equal to 'a'.
        Supported on fp16 operands only.

        Returns the ceil result.

        """