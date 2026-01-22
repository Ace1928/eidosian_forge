import logging
import io
import os
import shutil
import sys
import traceback
from contextlib import suppress
from enum import Enum
from inspect import cleandoc
from itertools import chain, starmap
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
from .. import (
from ..discovery import find_package_path
from ..dist import Distribution
from ..warnings import (
from .build_py import build_py as build_py_cls
import sys
from importlib.machinery import ModuleSpec, PathFinder
from importlib.machinery import all_suffixes as module_suffixes
from importlib.util import spec_from_file_location
from itertools import chain
from pathlib import Path
def _find_package_roots(packages: Iterable[str], package_dir: Mapping[str, str], src_root: _Path) -> Dict[str, str]:
    pkg_roots: Dict[str, str] = {pkg: _absolute_root(find_package_path(pkg, package_dir, src_root)) for pkg in sorted(packages)}
    return _remove_nested(pkg_roots)