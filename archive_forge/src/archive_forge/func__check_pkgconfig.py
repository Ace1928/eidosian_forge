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
def _check_pkgconfig(self, pkgbin: ExternalProgram) -> T.Optional[str]:
    if not pkgbin.found():
        mlog.log(f'Did not find pkg-config by name {pkgbin.name!r}')
        return None
    command_as_string = ' '.join(pkgbin.get_command())
    try:
        helptext = Popen_safe(pkgbin.get_command() + ['--help'])[1]
        if 'Pure-Perl' in helptext:
            mlog.log(f'Found pkg-config {command_as_string!r} but it is Strawberry Perl and thus broken. Ignoring...')
            return None
        p, out = Popen_safe(pkgbin.get_command() + ['--version'])[0:2]
        if p.returncode != 0:
            mlog.warning(f'Found pkg-config {command_as_string!r} but it failed when ran')
            return None
    except FileNotFoundError:
        mlog.warning(f"We thought we found pkg-config {command_as_string!r} but now it's not there. How odd!")
        return None
    except PermissionError:
        msg = f"Found pkg-config {command_as_string!r} but didn't have permissions to run it."
        if not self.env.machines.build.is_windows():
            msg += '\n\nOn Unix-like systems this is often caused by scripts that are not executable.'
        mlog.warning(msg)
        return None
    return out.strip()