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
def _looks_like_debian_scheme() -> bool:
    """Debian adds two additional schemes."""
    from distutils.command.install import INSTALL_SCHEMES
    return 'deb_system' in INSTALL_SCHEMES and 'unix_local' in INSTALL_SCHEMES