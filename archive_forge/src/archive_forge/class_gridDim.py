import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class gridDim(Dim3):
    """
    The shape of the grid of blocks. This value is the same for all threads in
    a given kernel launch.
    """
    _description_ = '<gridDim.{x,y,z}>'