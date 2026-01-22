import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
class LoweringError(NumbaError):
    """
    An error occurred during lowering.
    """

    def __init__(self, msg, loc=None):
        super(LoweringError, self).__init__(msg, loc=loc)