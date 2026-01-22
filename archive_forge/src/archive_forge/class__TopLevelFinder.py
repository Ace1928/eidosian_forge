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
class _TopLevelFinder:

    def __init__(self, dist: Distribution, name: str):
        self.dist = dist
        self.name = name

    def __call__(self, wheel: 'WheelFile', files: List[str], mapping: Dict[str, str]):
        src_root = self.dist.src_root or os.curdir
        top_level = chain(_find_packages(self.dist), _find_top_level_modules(self.dist))
        package_dir = self.dist.package_dir or {}
        roots = _find_package_roots(top_level, package_dir, src_root)
        namespaces_: Dict[str, List[str]] = dict(chain(_find_namespaces(self.dist.packages or [], roots), ((ns, []) for ns in _find_virtual_namespaces(roots))))
        legacy_namespaces = {pkg: find_package_path(pkg, roots, self.dist.src_root or '') for pkg in self.dist.namespace_packages or []}
        mapping = {**roots, **legacy_namespaces}
        name = f'__editable__.{self.name}.finder'
        finder = _normalization.safe_identifier(name)
        content = bytes(_finder_template(name, mapping, namespaces_), 'utf-8')
        wheel.writestr(f'{finder}.py', content)
        content = _encode_pth(f'import {finder}; {finder}.install()')
        wheel.writestr(f'__editable__.{self.name}.pth', content)

    def __enter__(self):
        msg = 'Editable install will be performed using a meta path finder.\n'
        _logger.warning(msg + _LENIENT_WARNING)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        msg = '\n\n        Please be careful with folders in your working directory with the same\n        name as your package as they may take precedence during imports.\n        '
        InformationOnly.emit('Editable installation.', msg)