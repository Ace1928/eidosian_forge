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
class NumbaPedanticWarning(NumbaWarning):
    """
    Warning category for reporting pedantic messages.
    """

    def __init__(self, msg, **kwargs):
        super().__init__(f'{msg}\n{pedantic_warning_info}')