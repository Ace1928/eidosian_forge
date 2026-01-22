import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _find_packages_within(root_pkg: str, pkg_dir: _Path) -> List[str]:
    nested = PEP420PackageFinder.find(pkg_dir)
    return [root_pkg] + ['.'.join((root_pkg, n)) for n in nested]