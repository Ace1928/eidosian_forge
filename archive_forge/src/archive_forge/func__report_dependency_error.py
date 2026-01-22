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
def _report_dependency_error(self, msg: str, ret_val: T.Optional[TV_ResultTuple]=None) -> T.Optional[TV_ResultTuple]:
    if self.required:
        raise DependencyException(msg)
    mlog.debug(msg)
    return ret_val