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
def _remove_nested(pkg_roots: Dict[str, str]) -> Dict[str, str]:
    output = dict(pkg_roots.copy())
    for pkg, path in reversed(list(pkg_roots.items())):
        if any((pkg != other and _is_nested(pkg, path, other, other_path) for other, other_path in pkg_roots.items())):
            output.pop(pkg)
    return output