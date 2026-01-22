import itertools
import os
from fnmatch import fnmatchcase
from glob import glob
from pathlib import Path
from typing import (
import _distutils_hack.override  # noqa: F401
from distutils import log
from distutils.util import convert_path
@property
def _package_dir(self) -> Dict[str, str]:
    if self.dist.package_dir is None:
        return {}
    return self.dist.package_dir