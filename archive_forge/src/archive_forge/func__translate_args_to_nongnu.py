from __future__ import annotations
import os.path
import re
import subprocess
import typing as T
from .. import mesonlib
from .. import mlog
from ..arglist import CompilerArgs
from ..linkers import RSPFileSyntax
from ..mesonlib import (
from . import compilers
from .compilers import (
from .mixins.gnu import GnuCompiler
from .mixins.gnu import gnu_common_warning_args
@classmethod
def _translate_args_to_nongnu(cls, args: T.List[str], info: MachineInfo, link_id: str) -> T.List[str]:
    dcargs: T.List[str] = []
    link_expect_arg = False
    link_flags_with_arg = ['-rpath', '-rpath-link', '-soname', '-compatibility_version', '-current_version']
    for arg in args:
        osargs: T.List[str] = []
        if info.is_windows():
            osargs = cls.translate_arg_to_windows(arg)
        elif info.is_darwin():
            osargs = cls._translate_arg_to_osx(arg)
        if osargs:
            dcargs.extend(osargs)
            continue
        if arg == '-pthread':
            continue
        if arg.startswith('-fstack-protector'):
            continue
        if arg.startswith('-D') and (not (arg == '-D' or arg.startswith(('-Dd', '-Df')))):
            continue
        if arg.startswith('-Wl,'):
            linkargs = arg[arg.index(',') + 1:].split(',')
            for la in linkargs:
                dcargs.append('-L=' + la.strip())
            continue
        elif arg.startswith(('-link-defaultlib', '-linker', '-link-internally', '-linkonce-templates', '-lib')):
            dcargs.append(arg)
            continue
        elif arg.startswith('-l'):
            dcargs.append('-L=' + arg)
            continue
        elif arg.startswith('-isystem'):
            if arg.startswith('-isystem='):
                dcargs.append('-I=' + arg[9:])
            else:
                dcargs.append('-I' + arg[8:])
            continue
        elif arg.startswith('-idirafter'):
            if arg.startswith('-idirafter='):
                dcargs.append('-I=' + arg[11:])
            else:
                dcargs.append('-I' + arg[10:])
            continue
        elif arg.startswith('-L'):
            if arg.startswith('-L='):
                suffix = arg[3:]
            else:
                suffix = arg[2:]
            if link_expect_arg:
                dcargs.append(arg)
                link_expect_arg = False
                continue
            if suffix in link_flags_with_arg:
                link_expect_arg = True
            if suffix.startswith('-') or suffix.startswith('@'):
                dcargs.append(arg)
                continue
            if info.is_windows() and link_id == 'link' and suffix.startswith('/'):
                dcargs.append(arg)
                continue
            if arg.endswith('.a') or arg.endswith('.lib'):
                if len(suffix) > 0 and (not suffix.startswith('-')):
                    dcargs.append('-L=' + suffix)
                    continue
            dcargs.append('-L=' + arg)
            continue
        elif not arg.startswith('-') and arg.endswith(('.a', '.lib')):
            dcargs.append('-L=' + arg)
            continue
        else:
            dcargs.append(arg)
    return dcargs