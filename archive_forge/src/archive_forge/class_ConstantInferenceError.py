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
class ConstantInferenceError(NumbaError):
    """
    Failure during constant inference.
    """

    def __init__(self, value, loc=None):
        super(ConstantInferenceError, self).__init__(value, loc=loc)