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
class EnsurePackagesDiscovered:
    """Some expand functions require all the packages to already be discovered before
    they run, e.g. :func:`read_attr`, :func:`resolve_class`, :func:`cmdclass`.

    Therefore in some cases we will need to run autodiscovery during the evaluation of
    the configuration. However, it is better to postpone calling package discovery as
    much as possible, because some parameters can influence it (e.g. ``package_dir``),
    and those might not have been processed yet.
    """

    def __init__(self, distribution: 'Distribution'):
        self._dist = distribution
        self._called = False

    def __call__(self):
        """Trigger the automatic package discovery, if it is still necessary."""
        if not self._called:
            self._called = True
            self._dist.set_defaults(name=False)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self._called:
            self._dist.set_defaults.analyse_name()

    def _get_package_dir(self) -> Mapping[str, str]:
        self()
        pkg_dir = self._dist.package_dir
        return {} if pkg_dir is None else pkg_dir

    @property
    def package_dir(self) -> Mapping[str, str]:
        """Proxy to ``package_dir`` that may trigger auto-discovery when used."""
        return LazyMappingProxy(self._get_package_dir)