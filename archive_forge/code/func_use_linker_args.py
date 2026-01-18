from __future__ import annotations
import abc
import functools
import os
import multiprocessing
import pathlib
import re
import subprocess
import typing as T
from ... import mesonlib
from ... import mlog
from ...mesonlib import OptionKey
from mesonbuild.compilers.compilers import CompileCheckMode
@classmethod
def use_linker_args(cls, linker: str, version: str) -> T.List[str]:
    if linker == 'mold' and mesonlib.version_compare(version, '>=12.0.1'):
        return ['-fuse-ld=mold']
    return super().use_linker_args(linker, version)