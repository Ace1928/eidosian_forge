import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def _deprecate_class(old_name, new_class, next_version, instancecheck=True):
    """
    Raise warning if a deprecated class is used in an isinstance check.
    """

    class _DeprecatedMeta(type):

        def __instancecheck__(self, other):
            warnings.warn(_DEPR_MSG.format(old_name, next_version, new_class.__name__), FutureWarning, stacklevel=2)
            return isinstance(other, new_class)
    return _DeprecatedMeta(old_name, (new_class,), {})