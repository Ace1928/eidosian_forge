import logging
import os
import sys
import sysconfig
import typing
from pip._internal.exceptions import InvalidSchemeCombination, UserInstallationInvalid
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import change_root, get_major_minor_version, is_osx_framework
def _infer_user() -> str:
    """Try to find a user scheme for the current platform."""
    if _PREFERRED_SCHEME_API:
        return _PREFERRED_SCHEME_API('user')
    if is_osx_framework() and (not running_under_virtualenv()):
        suffixed = 'osx_framework_user'
    else:
        suffixed = f'{os.name}_user'
    if suffixed in _AVAILABLE_SCHEMES:
        return suffixed
    if 'posix_user' not in _AVAILABLE_SCHEMES:
        raise UserInstallationInvalid()
    return 'posix_user'