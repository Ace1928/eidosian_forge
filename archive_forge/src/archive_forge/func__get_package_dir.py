import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def _get_package_dir(self) -> Mapping[str, str]:
    self()
    pkg_dir = self._dist.package_dir
    return {} if pkg_dir is None else pkg_dir