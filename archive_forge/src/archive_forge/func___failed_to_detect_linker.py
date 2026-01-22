from __future__ import annotations
from .. import mlog
from ..mesonlib import (
import re
import shlex
import typing as T
def __failed_to_detect_linker(compiler: T.List[str], args: T.List[str], stdout: str, stderr: str) -> 'T.NoReturn':
    msg = 'Unable to detect linker for compiler `{}`\nstdout: {}\nstderr: {}'.format(join_args(compiler + args), stdout, stderr)
    raise EnvironmentException(msg)