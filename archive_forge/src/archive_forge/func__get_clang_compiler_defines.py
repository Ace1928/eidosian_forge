from __future__ import annotations
from ..mesonlib import (
from ..envconfig import BinaryTable
from .. import mlog
from ..linkers import guess_win_linker, guess_nix_linker
import subprocess
import platform
import re
import shutil
import tempfile
import os
import typing as T
def _get_clang_compiler_defines(compiler: T.List[str]) -> T.Dict[str, str]:
    """
    Get the list of Clang pre-processor defines
    """
    args = compiler + ['-E', '-dM', '-']
    mlog.debug(f'Running command: {join_args(args)}')
    p, output, error = Popen_safe(args, write='', stdin=subprocess.PIPE)
    if p.returncode != 0:
        raise EnvironmentException(f'Unable to get clang pre-processor defines:\nCompiler stdout:\n{output}\n-----\nCompiler stderr:\n{error}\n-----\n')
    defines: T.Dict[str, str] = {}
    for line in output.split('\n'):
        if not line:
            continue
        d, *rest = line.split(' ', 2)
        if d != '#define':
            continue
        if len(rest) == 1:
            defines[rest[0]] = ''
        if len(rest) == 2:
            defines[rest[0]] = rest[1]
    return defines