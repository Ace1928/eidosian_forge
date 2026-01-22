import contextlib
import functools
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Type, cast
from pip._internal.utils.misc import strtobool
from .base import BaseDistribution, BaseEnvironment, FilesystemWheel, MemoryWheel, Wheel
def _should_use_importlib_metadata() -> bool:
    """Whether to use the ``importlib.metadata`` or ``pkg_resources`` backend.

    By default, pip uses ``importlib.metadata`` on Python 3.11+, and
    ``pkg_resourcess`` otherwise. This can be overridden by a couple of ways:

    * If environment variable ``_PIP_USE_IMPORTLIB_METADATA`` is set, it
      dictates whether ``importlib.metadata`` is used, regardless of Python
      version.
    * On Python 3.11+, Python distributors can patch ``importlib.metadata``
      to add a global constant ``_PIP_USE_IMPORTLIB_METADATA = False``. This
      makes pip use ``pkg_resources`` (unless the user set the aforementioned
      environment variable to *True*).
    """
    with contextlib.suppress(KeyError, ValueError):
        return bool(strtobool(os.environ['_PIP_USE_IMPORTLIB_METADATA']))
    if sys.version_info < (3, 11):
        return False
    import importlib.metadata
    return bool(getattr(importlib.metadata, '_PIP_USE_IMPORTLIB_METADATA', True))