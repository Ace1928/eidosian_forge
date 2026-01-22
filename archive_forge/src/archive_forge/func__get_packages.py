import collections
import os
import os.path
import subprocess
import sys
import sysconfig
import tempfile
from importlib import resources
import runpy
import sys
def _get_packages():
    global _PACKAGES, _WHEEL_PKG_DIR
    if _PACKAGES is not None:
        return _PACKAGES
    packages = {}
    for name, version, py_tag in _PROJECTS:
        wheel_name = f'{name}-{version}-{py_tag}-none-any.whl'
        packages[name] = _Package(version, wheel_name, None)
    if _WHEEL_PKG_DIR:
        dir_packages = _find_packages(_WHEEL_PKG_DIR)
        if all((name in dir_packages for name in _PACKAGE_NAMES)):
            packages = dir_packages
    _PACKAGES = packages
    return packages