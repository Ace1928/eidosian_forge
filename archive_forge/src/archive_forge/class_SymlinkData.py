from __future__ import annotations
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, InitVar
from functools import lru_cache
import abc
import hashlib
import itertools, pathlib
import os
import pickle
import re
import textwrap
import typing as T
from . import coredata
from . import dependencies
from . import mlog
from . import programs
from .mesonlib import (
from .compilers import (
from .interpreterbase import FeatureNew, FeatureDeprecated
@dataclass(eq=False)
class SymlinkData(HoldableObject):
    target: str
    name: str
    install_dir: str
    subproject: str
    install_tag: T.Optional[str] = None

    def __post_init__(self) -> None:
        if self.name != os.path.basename(self.name):
            raise InvalidArguments(f'Link name is "{self.name}", but link names cannot contain path separators. The dir part should be in install_dir.')