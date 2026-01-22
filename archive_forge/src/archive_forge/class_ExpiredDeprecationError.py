from __future__ import annotations
import functools
import re
import typing as ty
import warnings
class ExpiredDeprecationError(RuntimeError):
    """Error for expired deprecation

    Error raised when a called function or method has passed out of its
    deprecation period.
    """
    pass