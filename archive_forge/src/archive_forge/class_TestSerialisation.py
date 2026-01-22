from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
@dataclass(eq=False)
class TestSerialisation:
    name: str
    project_name: str
    suite: T.List[str]
    fname: T.List[str]
    is_cross_built: bool
    exe_wrapper: T.Optional[programs.ExternalProgram]
    needs_exe_wrapper: bool
    is_parallel: bool
    cmd_args: T.List[str]
    env: mesonlib.EnvironmentVariables
    should_fail: bool
    timeout: T.Optional[int]
    workdir: T.Optional[str]
    extra_paths: T.List[str]
    protocol: TestProtocol
    priority: int
    cmd_is_built: bool
    cmd_is_exe: bool
    depends: T.List[str]
    version: str
    verbose: bool

    def __post_init__(self) -> None:
        if self.exe_wrapper is not None:
            assert isinstance(self.exe_wrapper, programs.ExternalProgram)