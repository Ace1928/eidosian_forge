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
class NumbaPendingDeprecationWarning(NumbaWarning, PendingDeprecationWarning):
    """
    Warning category for use of a feature that is pending deprecation.
    """