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
def _collect_build_outputs(self) -> Tuple[List[str], Dict[str, str]]:
    files: List[str] = []
    mapping: Dict[str, str] = {}
    build = self.get_finalized_command('build')
    for cmd_name in build.get_sub_commands():
        cmd = self.get_finalized_command(cmd_name)
        if hasattr(cmd, 'get_outputs'):
            files.extend(cmd.get_outputs() or [])
        if hasattr(cmd, 'get_output_mapping'):
            mapping.update(cmd.get_output_mapping() or {})
    return (files, mapping)