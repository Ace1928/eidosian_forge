from __future__ import annotations
from .common import cmake_is_debug
from .. import mlog
from ..mesonlib import Version
from pathlib import Path
import re
import typing as T
class ResolvedTarget:

    def __init__(self) -> None:
        self.include_directories: T.List[str] = []
        self.link_flags: T.List[str] = []
        self.public_compile_opts: T.List[str] = []
        self.libraries: T.List[str] = []