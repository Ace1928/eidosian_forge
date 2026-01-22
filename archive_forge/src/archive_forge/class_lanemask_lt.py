import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class lanemask_lt(Stub):
    """
    lanemask_lt()

    Returns a 32-bit integer mask of all lanes (including inactive ones) with
    ID less than the current lane.
    """
    _description_ = '<lanemask_lt()>'