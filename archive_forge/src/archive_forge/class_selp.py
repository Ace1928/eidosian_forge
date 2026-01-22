import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class selp(Stub):
    """
    selp(a, b, c)

    Select between source operands, based on the value of the predicate source
    operand.
    """