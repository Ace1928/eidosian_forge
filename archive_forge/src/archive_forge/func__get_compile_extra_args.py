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
def _get_compile_extra_args(self, extra_args: T.Union[T.List[str], T.Callable[[CompileCheckMode], T.List[str]], None]=None) -> T.List[str]:
    args = self._get_target_arch_args()
    if extra_args:
        if callable(extra_args):
            extra_args = extra_args(CompileCheckMode.COMPILE)
        if isinstance(extra_args, list):
            args.extend(extra_args)
        elif isinstance(extra_args, str):
            args.append(extra_args)
    return args