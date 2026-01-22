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
class InternalTargetMismatchError(InternalError):
    """For signalling a target mismatch error occurred internally within the
    compiler.
    """

    def __init__(self, kind, target_hw, hw_clazz):
        msg = f'{kind.title()} being resolved on a target from which it does not inherit. Local target is {target_hw}, declared target class is {hw_clazz}.'
        super().__init__(msg)