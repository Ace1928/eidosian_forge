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
class NumbaDebugInfoWarning(NumbaWarning):
    """
    Warning category for an issue with the emission of debug information.
    """