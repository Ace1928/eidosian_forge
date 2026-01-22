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
def _run_build_commands(self, dist_name: str, unpacked_wheel: _Path, build_lib: _Path, tmp_dir: _Path) -> Tuple[List[str], Dict[str, str]]:
    self._configure_build(dist_name, unpacked_wheel, build_lib, tmp_dir)
    self._run_build_subcommands()
    files, mapping = self._collect_build_outputs()
    self._run_install('headers')
    self._run_install('scripts')
    self._run_install('data')
    return (files, mapping)