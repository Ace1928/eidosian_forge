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
def update_vers(new_vers: str) -> None:
    nonlocal cur_vers
    new_vers = new_vers.replace('_', '')
    new_vers = new_vers.replace('.', '')
    if not new_vers.isdigit():
        return
    if len(new_vers) > len(cur_vers):
        cur_vers = new_vers