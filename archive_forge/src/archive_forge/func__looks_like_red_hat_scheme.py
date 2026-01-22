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
def _looks_like_red_hat_scheme() -> bool:
    """Red Hat patches ``sys.prefix`` and ``sys.exec_prefix``.

    Red Hat's ``00251-change-user-install-location.patch`` changes the install
    command's ``prefix`` and ``exec_prefix`` to append ``"/local"``. This is
    (fortunately?) done quite unconditionally, so we create a default command
    object without any configuration to detect this.
    """
    from distutils.command.install import install
    from distutils.dist import Distribution
    cmd: Any = install(Distribution())
    cmd.finalize_options()
    return cmd.exec_prefix == f'{os.path.normpath(sys.exec_prefix)}/local' and cmd.prefix == f'{os.path.normpath(sys.prefix)}/local'