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
def _looks_like_red_hat_patched_platlib_purelib(scheme: Dict[str, str]) -> bool:
    platlib = scheme['platlib']
    if '/$platlibdir/' in platlib:
        platlib = platlib.replace('/$platlibdir/', f'/{_PLATLIBDIR}/')
    if '/lib64/' not in platlib:
        return False
    unpatched = platlib.replace('/lib64/', '/lib/')
    return unpatched.replace('$platbase/', '$base/') == scheme['purelib']