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
class ByteCodeSupportError(NumbaError):
    """
    Failure to extract the bytecode of the user's function.
    """

    def __init__(self, msg, loc=None):
        super(ByteCodeSupportError, self).__init__(msg, loc=loc)