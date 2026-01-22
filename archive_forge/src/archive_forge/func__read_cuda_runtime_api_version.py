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
def _read_cuda_runtime_api_version(self, path_str: str) -> T.Optional[str]:
    path = Path(path_str)
    for i in path.rglob('cuda_runtime_api.h'):
        raw = i.read_text(encoding='utf-8')
        m = self.cudart_version_regex.search(raw)
        if not m:
            continue
        try:
            vers_int = int(m.group(1))
        except ValueError:
            continue
        major = vers_int // 1000
        minor = (vers_int - major * 1000) // 10
        return f'{major}.{minor}'
    return None