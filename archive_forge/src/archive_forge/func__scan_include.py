from __future__ import annotations
import copy
import itertools
import functools
import os
import subprocess
import textwrap
import typing as T
from . import (
from .. import build
from .. import interpreter
from .. import mesonlib
from .. import mlog
from ..build import CustomTarget, CustomTargetIndex, Executable, GeneratedList, InvalidArguments
from ..dependencies import Dependency, InternalDependency
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import DEPENDS_KW, DEPEND_FILES_KW, ENV_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, DEPENDENCY_SOURCES_KW, in_set_validator
from ..interpreterbase import noPosargs, noKwargs, FeatureNew, FeatureDeprecated
from ..interpreterbase import typed_kwargs, KwargInfo, ContainerTypeInfo
from ..interpreterbase.decorators import typed_pos_args
from ..mesonlib import (
from ..programs import OverrideProgram
from ..scripts.gettext import read_linguas
@staticmethod
def _scan_include(state: 'ModuleState', includes: T.List[T.Union[str, GirTarget]]) -> T.Tuple[T.List[str], T.List[str], T.List[GirTarget]]:
    ret: T.List[str] = []
    gir_inc_dirs: T.List[str] = []
    depends: T.List[GirTarget] = []
    for inc in includes:
        if isinstance(inc, str):
            ret += [f'--include={inc}']
        elif isinstance(inc, GirTarget):
            gir_inc_dirs.append(os.path.join(state.environment.get_build_dir(), inc.get_subdir()))
            ret.append(f'--include-uninstalled={os.path.join(inc.get_subdir(), inc.get_basename())}')
            depends.append(inc)
    return (ret, gir_inc_dirs, depends)