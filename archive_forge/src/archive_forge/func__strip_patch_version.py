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
@classmethod
def _strip_patch_version(cls, version: str) -> str:
    return '.'.join(version.split('.')[:2])