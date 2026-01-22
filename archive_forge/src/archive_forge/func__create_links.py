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
def _create_links(self, outputs, output_mapping):
    self.auxiliary_dir.mkdir(parents=True, exist_ok=True)
    link_type = 'sym' if _can_symlink_files(self.auxiliary_dir) else 'hard'
    mappings = {self._normalize_output(k): v for k, v in output_mapping.items()}
    mappings.pop(None, None)
    for output in outputs:
        relative = self._normalize_output(output)
        if relative and relative not in mappings:
            self._create_file(relative, output)
    for relative, src in mappings.items():
        self._create_file(relative, src, link=link_type)