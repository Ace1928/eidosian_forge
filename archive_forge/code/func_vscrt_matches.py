from __future__ import annotations
import re
import dataclasses
import functools
import typing as T
from pathlib import Path
from .. import mlog
from .. import mesonlib
from .base import DependencyException, SystemDependency
from .detect import packages
from .pkgconfig import PkgConfigDependency
from .misc import threads_factory
def vscrt_matches(self, vscrt: str) -> bool:
    if not vscrt:
        return True
    if vscrt in {'/MD', '-MD'}:
        return not self.runtime_static and (not self.runtime_debug)
    elif vscrt in {'/MDd', '-MDd'}:
        return not self.runtime_static and self.runtime_debug
    elif vscrt in {'/MT', '-MT'}:
        return (self.runtime_static or not self.static) and (not self.runtime_debug)
    elif vscrt in {'/MTd', '-MTd'}:
        return (self.runtime_static or not self.static) and self.runtime_debug
    mlog.warning(f'Boost: unknown vscrt tag {vscrt}. This may cause the compilation to fail. Please consider reporting this as a bug.', once=True)
    return True