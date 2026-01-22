import functools
import logging
import os
import pathlib
import sys
import sysconfig
from typing import Any, Dict, Generator, Optional, Tuple
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.virtualenv import running_under_virtualenv
from . import _sysconfig
from .base import (
@functools.lru_cache(maxsize=None)
def _looks_like_red_hat_lib() -> bool:
    """Red Hat patches platlib in unix_prefix and unix_home, but not purelib.

    This is the only way I can see to tell a Red Hat-patched Python.
    """
    from distutils.command.install import INSTALL_SCHEMES
    return all((k in INSTALL_SCHEMES and _looks_like_red_hat_patched_platlib_purelib(INSTALL_SCHEMES[k]) for k in ('unix_prefix', 'unix_home')))