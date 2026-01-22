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
@staticmethod
def _get_id_hash(target_id: str) -> str:
    h = hashlib.sha256()
    h.update(target_id.encode(encoding='utf-8', errors='replace'))
    return h.hexdigest()[:7]