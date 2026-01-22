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
def _expand_cmdclass(self, package_dir: Mapping[str, str]):
    root_dir = self.root_dir
    cmdclass = partial(_expand.cmdclass, package_dir=package_dir, root_dir=root_dir)
    self._process_field(self.setuptools_cfg, 'cmdclass', cmdclass)