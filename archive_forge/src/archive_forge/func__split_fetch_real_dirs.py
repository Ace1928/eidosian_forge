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
def _split_fetch_real_dirs(self, pathstr: str) -> T.List[str]:
    pathsep = os.pathsep
    if pathsep == ';':
        pathstr = re.sub(':([^/\\\\])', ';\\1', pathstr)
    paths = [p for p in pathstr.split(pathsep) if p]
    result: T.List[str] = []
    for p in paths:
        pobj = pathlib.Path(p)
        unresolved = pobj.as_posix()
        if pobj.exists():
            if unresolved not in result:
                result.append(unresolved)
            try:
                resolved = pathlib.Path(p).resolve().as_posix()
                if resolved not in result:
                    result.append(resolved)
            except FileNotFoundError:
                pass
    return result