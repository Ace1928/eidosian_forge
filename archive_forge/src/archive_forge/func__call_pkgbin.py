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
def _call_pkgbin(self, args: T.List[str], env: T.Optional[EnvironOrDict]=None) -> T.Tuple[int, str, str]:
    assert isinstance(self.pkgbin, ExternalProgram)
    env = env or os.environ
    env = self._setup_env(env)
    cmd = self.pkgbin.get_command() + args
    p, out, err = Popen_safe_logged(cmd, env=env)
    return (p.returncode, out.strip(), err.strip())