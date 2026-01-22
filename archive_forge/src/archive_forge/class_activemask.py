import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class activemask(Stub):
    """
    activemask()

    Returns a 32-bit integer mask of all currently active threads in the
    calling warp. The Nth bit is set if the Nth lane in the warp is active when
    activemask() is called. Inactive threads are represented by 0 bits in the
    returned mask. Threads which have exited the kernel are always marked as
    inactive.
    """
    _description_ = '<activemask()>'