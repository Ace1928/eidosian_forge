from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
@staticmethod
def _quoter(x, qf=quote_func):
    if isinstance(x, NinjaCommandArg):
        if x.quoting == Quoting.none:
            return x.s
        elif x.quoting == Quoting.notNinja:
            return qf(x.s)
        elif x.quoting == Quoting.notShell:
            return ninja_quote(x.s)
    return ninja_quote(qf(str(x)))