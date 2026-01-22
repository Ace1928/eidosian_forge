import logging
import platform
import subprocess
import sys
import sysconfig
from importlib.machinery import EXTENSION_SUFFIXES
from typing import (
from . import _manylinux, _musllinux
def _cpython_abis(py_version: PythonVersion, warn: bool=False) -> List[str]:
    py_version = tuple(py_version)
    abis = []
    version = _version_nodot(py_version[:2])
    debug = pymalloc = ucs4 = ''
    with_debug = _get_config_var('Py_DEBUG', warn)
    has_refcount = hasattr(sys, 'gettotalrefcount')
    has_ext = '_d.pyd' in EXTENSION_SUFFIXES
    if with_debug or (with_debug is None and (has_refcount or has_ext)):
        debug = 'd'
    if py_version < (3, 8):
        with_pymalloc = _get_config_var('WITH_PYMALLOC', warn)
        if with_pymalloc or with_pymalloc is None:
            pymalloc = 'm'
        if py_version < (3, 3):
            unicode_size = _get_config_var('Py_UNICODE_SIZE', warn)
            if unicode_size == 4 or (unicode_size is None and sys.maxunicode == 1114111):
                ucs4 = 'u'
    elif debug:
        abis.append(f'cp{version}')
    abis.insert(0, 'cp{version}{debug}{pymalloc}{ucs4}'.format(version=version, debug=debug, pymalloc=pymalloc, ucs4=ucs4))
    return abis