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
def _cuda_paths_win(self) -> T.List[T.Tuple[str, bool]]:
    env_vars = os.environ.keys()
    return [(os.environ[var], False) for var in env_vars if var.startswith('CUDA_PATH_')]