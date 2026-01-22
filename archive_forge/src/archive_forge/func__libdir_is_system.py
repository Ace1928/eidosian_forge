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
@staticmethod
def _libdir_is_system(libdir: str, compilers: T.Mapping[str, 'Compiler'], env: 'Environment') -> bool:
    libdir = os.path.normpath(libdir)
    for cc in compilers.values():
        if libdir in cc.get_library_dirs(env):
            return True
    return False