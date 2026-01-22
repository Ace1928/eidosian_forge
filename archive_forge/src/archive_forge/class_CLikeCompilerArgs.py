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
class CLikeCompilerArgs(arglist.CompilerArgs):
    prepend_prefixes = ('-I', '-L')
    dedup2_prefixes = ('-I', '-isystem', '-L', '-D', '-U')
    dedup1_prefixes = ('-l', '-Wl,-l', '-Wl,--export-dynamic')
    dedup1_suffixes = ('.lib', '.dll', '.so', '.dylib', '.a')
    dedup1_args = ('-c', '-S', '-E', '-pipe', '-pthread')

    def to_native(self, copy: bool=False) -> T.List[str]:
        assert isinstance(self.compiler, compilers.Compiler), 'How did you get here'
        self.flush_pre_post()
        if copy:
            new = self.copy()
        else:
            new = self
        if isinstance(self.compiler.linker, (GnuLikeDynamicLinkerMixin, SolarisDynamicLinker, CompCertDynamicLinker)):
            group_start = -1
            group_end = -1
            for i, each in enumerate(new):
                if not GROUP_FLAGS.search(each):
                    continue
                group_end = i
                if group_start < 0:
                    group_start = i
            if group_end > group_start >= 0:
                new.insert(group_end + 1, '-Wl,--end-group')
                new.insert(group_start, '-Wl,--start-group')
        default_dirs = self.compiler.get_default_include_dirs()
        if default_dirs:
            real_default_dirs = [self._cached_realpath(i) for i in default_dirs]
            bad_idx_list: T.List[int] = []
            for i, each in enumerate(new):
                if not each.startswith('-isystem'):
                    continue
                if each == '-isystem' and i < len(new) - 1 and (self._cached_realpath(new[i + 1]) in real_default_dirs):
                    bad_idx_list += [i, i + 1]
                elif each.startswith('-isystem=') and self._cached_realpath(each[9:]) in real_default_dirs:
                    bad_idx_list += [i]
                elif self._cached_realpath(each[8:]) in real_default_dirs:
                    bad_idx_list += [i]
            for i in reversed(bad_idx_list):
                new.pop(i)
        return self.compiler.unix_args_to_native(new._container)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _cached_realpath(arg: str) -> str:
        return os.path.realpath(arg)

    def __repr__(self) -> str:
        self.flush_pre_post()
        return f'CLikeCompilerArgs({self.compiler!r}, {self._container!r})'