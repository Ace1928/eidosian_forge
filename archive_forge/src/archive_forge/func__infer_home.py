import logging
import os
import sys
import sysconfig
import typing
from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import change_root, get_major_minor_version, is_osx_framework
def _infer_home() -> str:
    """Try to find a home for the current platform."""
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API('home')
    suffixed = f'{os.name}_home'
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    return 'posix_home'