import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class habs(Stub):
    """habs(a)

        Perform fp16 absolute value, |a|. Supported on fp16 operands only.

        Returns the fp16 result of the absolute value.

        """