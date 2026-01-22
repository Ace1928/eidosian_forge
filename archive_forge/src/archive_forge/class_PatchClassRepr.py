import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class PatchClassRepr(type):
    """Control class representations in rendered signatures."""

    def __repr__(cls):
        return f'<{cls.__name__}>'