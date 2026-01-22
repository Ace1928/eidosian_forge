import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
@staticmethod
def _looks_like_package(_path: _Path, package_name: str) -> bool:
    names = package_name.split('.')
    root_pkg_is_valid = names[0].isidentifier() or names[0].endswith('-stubs')
    return root_pkg_is_valid and all((name.isidentifier() for name in names[1:]))