from __future__ import annotations
import collections
import functools
import glob
import itertools
import os
import re
import subprocess
import copy
import typing as T
from pathlib import Path
from ... import arglist
from ... import mesonlib
from ... import mlog
from ...linkers.linkers import GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker
from ...mesonlib import LibType, OptionKey
from .. import compilers
from ..compilers import CompileCheckMode
from .visualstudio import VisualStudioLikeCompiler
def _has_multi_arguments(self, args: T.List[str], env: 'Environment', code: str) -> T.Tuple[bool, bool]:
    new_args: T.List[str] = []
    for arg in args:
        if arg.startswith('-Wno-'):
            new_args.append('-W' + arg[5:])
        if arg.startswith('-Wl,'):
            mlog.warning(f'{arg} looks like a linker argument, but has_argument and other similar methods only support checking compiler arguments. Using them to check linker arguments are never supported, and results are likely to be wrong regardless of the compiler you are using. has_link_argument or other similar method can be used instead.')
        new_args.append(arg)
    return self.has_arguments(new_args, env, code, mode=CompileCheckMode.COMPILE)