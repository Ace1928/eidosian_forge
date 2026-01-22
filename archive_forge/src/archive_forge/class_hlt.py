import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class hlt(Stub):
    """hlt(a, b)

        Perform fp16 comparison, (a < b). Supported
        on fp16 operands only.

        Returns True if a is < b and False otherwise.

        """