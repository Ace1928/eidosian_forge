import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Mapping, Optional, Set, Union
from ..errors import FileError, InvalidConfigError
from ..warnings import SetuptoolsWarning
from . import expand as _expand
from ._apply_pyprojecttoml import _PREVIOUSLY_DEFINED, _MissingDynamic
from ._apply_pyprojecttoml import apply as _apply
def _expand_packages(self):
    packages = self.setuptools_cfg.get('packages')
    if packages is None or isinstance(packages, (list, tuple)):
        return
    find = packages.get('find')
    if isinstance(find, dict):
        find['root_dir'] = self.root_dir
        find['fill_package_dir'] = self.setuptools_cfg.setdefault('package-dir', {})
        with _ignore_errors(self.ignore_option_errors):
            self.setuptools_cfg['packages'] = _expand.find_packages(**find)