import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
def _analyse_flat_packages(self) -> bool:
    self.dist.packages = FlatLayoutPackageFinder.find(self._root_dir)
    top_level = remove_nested_packages(remove_stubs(self.dist.packages))
    log.debug(f'discovered packages -- {self.dist.packages}')
    self._ensure_no_accidental_inclusion(top_level, 'packages')
    return bool(top_level)