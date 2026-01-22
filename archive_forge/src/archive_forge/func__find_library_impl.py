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
def _find_library_impl(self, libname: str, env: 'Environment', extra_dirs: T.List[str], code: str, libtype: LibType, lib_prefix_warning: bool) -> T.Optional[T.List[str]]:
    if libname in self.ignore_libs:
        return []
    if isinstance(extra_dirs, str):
        extra_dirs = [extra_dirs]
    key = (tuple(self.exelist), libname, tuple(extra_dirs), code, libtype)
    if key not in self.find_library_cache:
        value = self._find_library_real(libname, env, extra_dirs, code, libtype, lib_prefix_warning)
        self.find_library_cache[key] = value
    else:
        value = self.find_library_cache[key]
    if value is None:
        return None
    return value.copy()