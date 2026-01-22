import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class nanosleep(Stub):
    """
    nanosleep(ns)

    Suspends the thread for a sleep duration approximately close to the delay
    `ns`, specified in nanoseconds.
    """
    _description_ = '<nansleep()>'