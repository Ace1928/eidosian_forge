import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class match_any_sync(Stub):
    """
    match_any_sync(mask, value)

    Nvvm intrinsic for performing a compare and broadcast across a warp.
    Returns a mask of threads that have same value as the given value from
    within the masked warp.
    """
    _description_ = '<match_any_sync()>'