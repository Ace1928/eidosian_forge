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
def _detect_language(cls, compilers: T.Dict[str, 'Compiler']) -> str:
    for lang in cls.supported_languages:
        if lang in compilers:
            return lang
    return list(compilers.keys())[0]