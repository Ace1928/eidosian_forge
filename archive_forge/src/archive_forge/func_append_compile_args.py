from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
@typed_pos_args('subproject_options.append_compile_args', str, varargs=str, min_varargs=1)
@permittedKwargs({'target'})
def append_compile_args(self, state: ModuleState, args: T.Tuple[str, T.List[str]], kwargs: TYPE_kwargs) -> None:
    self._get_opts(kwargs).append_args(args[0], args[1])