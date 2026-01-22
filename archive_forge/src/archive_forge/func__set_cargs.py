from __future__ import annotations
from pathlib import Path
from .base import ExternalDependency, DependencyException, sort_libpaths, DependencyTypeName
from ..mesonlib import EnvironmentVariables, OptionKey, OrderedSet, PerMachine, Popen_safe, Popen_safe_logged, MachineChoice, join_args
from ..programs import find_external_program, ExternalProgram
from .. import mlog
from pathlib import PurePath
from functools import lru_cache
import re
import os
import shlex
import typing as T
def _set_cargs(self) -> None:
    allow_system = False
    if self.language == 'fortran':
        allow_system = True
    cflags = self.pkgconfig.cflags(self.name, allow_system)
    self.compile_args = self._convert_mingw_paths(cflags)