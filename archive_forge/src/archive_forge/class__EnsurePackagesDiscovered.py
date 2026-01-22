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
class _EnsurePackagesDiscovered(_expand.EnsurePackagesDiscovered):

    def __init__(self, distribution: 'Distribution', project_cfg: dict, setuptools_cfg: dict):
        super().__init__(distribution)
        self._project_cfg = project_cfg
        self._setuptools_cfg = setuptools_cfg

    def __enter__(self):
        """When entering the context, the values of ``packages``, ``py_modules`` and
        ``package_dir`` that are missing in ``dist`` are copied from ``setuptools_cfg``.
        """
        dist, cfg = (self._dist, self._setuptools_cfg)
        package_dir: Dict[str, str] = cfg.setdefault('package-dir', {})
        package_dir.update(dist.package_dir or {})
        dist.package_dir = package_dir
        dist.set_defaults._ignore_ext_modules()
        if dist.metadata.name is None:
            dist.metadata.name = self._project_cfg.get('name')
        if dist.py_modules is None:
            dist.py_modules = cfg.get('py-modules')
        if dist.packages is None:
            dist.packages = cfg.get('packages')
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """When exiting the context, if values of ``packages``, ``py_modules`` and
        ``package_dir`` are missing in ``setuptools_cfg``, copy from ``dist``.
        """
        self._setuptools_cfg.setdefault('packages', self._dist.packages)
        self._setuptools_cfg.setdefault('py-modules', self._dist.py_modules)
        return super().__exit__(exc_type, exc_value, traceback)