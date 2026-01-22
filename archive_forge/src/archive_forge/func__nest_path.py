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
def _nest_path(parent: _Path, path: _Path) -> str:
    path = parent if path in {'.', ''} else os.path.join(parent, path)
    return os.path.normpath(path)