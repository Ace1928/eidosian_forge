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
def validate_sources(self):
    if len(self.compilers) > 1 and any((lang in self.compilers for lang in ['cs', 'java'])):
        langs = ', '.join(self.compilers.keys())
        raise InvalidArguments(f'Cannot mix those languages into a target: {langs}')