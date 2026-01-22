from __future__ import annotations
import abc
import re
import os
import typing as T
from .base import DependencyException, DependencyMethods
from .configtool import ConfigToolDependency
from .detect import packages
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
from .factory import DependencyFactory
from .. import mlog
from .. import mesonlib
def _get_modules_lib_suffix(version: str, info: 'MachineInfo', is_debug: bool) -> str:
    """Get the module suffix based on platform and debug type."""
    suffix = ''
    if info.is_windows():
        if is_debug:
            suffix += 'd'
        if version.startswith('4'):
            suffix += '4'
    if info.is_darwin():
        if is_debug:
            suffix += '_debug'
    if mesonlib.version_compare(version, '>= 5.14.0'):
        if info.is_android():
            if info.cpu_family == 'x86':
                suffix += '_x86'
            elif info.cpu_family == 'x86_64':
                suffix += '_x86_64'
            elif info.cpu_family == 'arm':
                suffix += '_armeabi-v7a'
            elif info.cpu_family == 'aarch64':
                suffix += '_arm64-v8a'
            else:
                mlog.warning(f'Android target arch "{info.cpu_family}"" for Qt5 is unknown, module detection may not work')
    return suffix