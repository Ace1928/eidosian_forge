import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class warpsize(Stub):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """
    _description_ = '<warpsize>'