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
def _find_module(module_name: str, package_dir: Optional[Mapping[str, str]], root_dir: _Path) -> Tuple[_Path, Optional[str], str]:
    """Given a module (that could normally be imported by ``module_name``
    after the build is complete), find the path to the parent directory where
    it is contained and the canonical name that could be used to import it
    considering the ``package_dir`` in the build configuration and ``root_dir``
    """
    parent_path = root_dir
    module_parts = module_name.split('.')
    if package_dir:
        if module_parts[0] in package_dir:
            custom_path = package_dir[module_parts[0]]
            parts = custom_path.rsplit('/', 1)
            if len(parts) > 1:
                parent_path = os.path.join(root_dir, parts[0])
                parent_module = parts[1]
            else:
                parent_module = custom_path
            module_name = '.'.join([parent_module, *module_parts[1:]])
        elif '' in package_dir:
            parent_path = os.path.join(root_dir, package_dir[''])
    path_start = os.path.join(parent_path, *module_name.split('.'))
    candidates = chain((f'{path_start}.py', os.path.join(path_start, '__init__.py')), iglob(f'{path_start}.*'))
    module_path = next((x for x in candidates if os.path.isfile(x)), None)
    return (parent_path, module_path, module_name)