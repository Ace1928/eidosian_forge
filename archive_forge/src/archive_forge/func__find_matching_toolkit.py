from __future__ import annotations
import glob
import re
import os
import typing as T
from pathlib import Path
from .. import mesonlib
from .. import mlog
from ..environment import detect_cpu_family
from .base import DependencyException, SystemDependency
from .detect import packages
def _find_matching_toolkit(self, paths: T.List[TV_ResultTuple], version_reqs: T.List[str], nvcc_version: T.Optional[str]) -> TV_ResultTuple:
    part_func: T.Callable[[TV_ResultTuple], bool] = lambda t: not t[2]
    defaults_it, rest_it = mesonlib.partition(part_func, paths)
    defaults = list(defaults_it)
    paths = defaults + sorted(rest_it, key=lambda t: mesonlib.Version(t[1]), reverse=True)
    mlog.debug(f'Search paths: {paths}')
    if nvcc_version and defaults:
        default_src = f'the {self.env_var} environment variable' if self.env_var else "the '/usr/local/cuda' symbolic link"
        nvcc_warning = "The default CUDA Toolkit as designated by {} ({}) doesn't match the current nvcc version {} and will be ignored.".format(default_src, os.path.realpath(defaults[0][0]), nvcc_version)
    else:
        nvcc_warning = None
    for path, version, default in paths:
        found_some, not_found, found = mesonlib.version_compare_many(version, version_reqs)
        if not not_found:
            if not default and nvcc_warning:
                mlog.warning(nvcc_warning)
            return (path, version, True)
    if nvcc_warning:
        mlog.warning(nvcc_warning)
    return (None, None, False)