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
def _convert_mingw_paths(self, args: ImmutableListProtocol[str]) -> T.List[str]:
    """
        Both MSVC and native Python on Windows cannot handle MinGW-esque /c/foo
        paths so convert them to C:/foo. We cannot resolve other paths starting
        with / like /home/foo so leave them as-is so that the user gets an
        error/warning from the compiler/linker.
        """
    if not self.env.machines.build.is_windows():
        return args.copy()
    converted = []
    for arg in args:
        pargs: T.Tuple[str, ...] = tuple()
        if arg.startswith('-L/'):
            pargs = PurePath(arg[2:]).parts
            tmpl = '-L{}:/{}'
        elif arg.startswith('-I/'):
            pargs = PurePath(arg[2:]).parts
            tmpl = '-I{}:/{}'
        elif arg.startswith('/'):
            pargs = PurePath(arg).parts
            tmpl = '{}:/{}'
        elif arg.startswith(('-L', '-I')) or (len(arg) > 2 and arg[1] == ':'):
            arg = arg.replace('\\ ', ' ')
        if len(pargs) > 1 and len(pargs[1]) == 1:
            arg = tmpl.format(pargs[1], '/'.join(pargs[2:]))
        converted.append(arg)
    return converted