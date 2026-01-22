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
def _expand_all_dynamic(self, dist: 'Distribution', package_dir: Mapping[str, str]):
    special = ('version', 'readme', 'entry-points', 'scripts', 'gui-scripts', 'classifiers', 'dependencies', 'optional-dependencies')
    obtained_dynamic = {field: self._obtain(dist, field, package_dir) for field in self.dynamic if field not in special}
    obtained_dynamic.update(self._obtain_entry_points(dist, package_dir) or {}, version=self._obtain_version(dist, package_dir), readme=self._obtain_readme(dist), classifiers=self._obtain_classifiers(dist), dependencies=self._obtain_dependencies(dist), optional_dependencies=self._obtain_optional_dependencies(dist))
    updates = {k: v for k, v in obtained_dynamic.items() if v is not None}
    self.project_cfg.update(updates)