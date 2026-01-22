import ast
import importlib
import os
import pathlib
import sys
from glob import iglob
from configparser import ConfigParser
from importlib.machinery import ModuleSpec
from itertools import chain
from typing import (
from pathlib import Path
from types import ModuleType
from distutils.errors import DistutilsOptionError
from .._path import same_path as _same_path
from ..warnings import SetuptoolsWarning
def _find_assignments(self) -> Iterator[Tuple[ast.AST, ast.AST]]:
    for statement in self.module.body:
        if isinstance(statement, ast.Assign):
            yield from ((target, statement.value) for target in statement.targets)
        elif isinstance(statement, ast.AnnAssign) and statement.value:
            yield (statement.target, statement.value)