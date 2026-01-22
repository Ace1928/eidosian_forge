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
@classmethod
def _get_trials_from_pattern(cls, pattern: str, directory: str, libname: str) -> T.List[Path]:
    f = Path(directory) / pattern.format(libname)
    if '*' in pattern:
        return [Path(x) for x in cls._sort_shlibs_openbsd(glob.glob(str(f)))]
    return [f]