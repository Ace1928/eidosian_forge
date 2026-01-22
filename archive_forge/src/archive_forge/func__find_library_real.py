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
def _find_library_real(self, libname: str, env: 'Environment', extra_dirs: T.List[str], code: str, libtype: LibType, lib_prefix_warning: bool) -> T.Optional[T.List[str]]:
    if not extra_dirs and libtype is LibType.PREFER_SHARED or libname in self.internal_libs:
        cargs = ['-l' + libname]
        largs = self.get_linker_always_args() + self.get_allow_undefined_link_args()
        extra_args = cargs + self.linker_to_compiler_args(largs)
        if self.links(code, env, extra_args=extra_args, disable_cache=True)[0]:
            return cargs
        if libname in self.internal_libs:
            return None
    patterns = self.get_library_naming(env, libtype)
    try:
        if self.output_is_64bit(env):
            elf_class = 2
        else:
            elf_class = 1
    except (mesonlib.MesonException, KeyError):
        elf_class = 0
    for d in itertools.chain(extra_dirs, self.get_library_dirs(env, elf_class)):
        for p in patterns:
            trials = self._get_trials_from_pattern(p, d, libname)
            if not trials:
                continue
            trial = self._get_file_from_list(env, trials)
            if not trial:
                continue
            if libname.startswith('lib') and trial.name.startswith(libname) and lib_prefix_warning:
                mlog.warning(f'find_library({libname!r}) starting in "lib" only works by accident and is not portable')
            return [trial.as_posix()]
    return None