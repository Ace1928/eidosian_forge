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
def _make_typelib_target(state: 'ModuleState', typelib_output: str, typelib_cmd: T.Sequence[T.Union[str, Executable, ExternalProgram, CustomTarget]], generated_files: T.Sequence[T.Union[str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList]], kwargs: T.Dict[str, T.Any]) -> TypelibTarget:
    install = kwargs['install_typelib']
    if install is None:
        install = kwargs['install']
    install_dir = kwargs['install_dir_typelib']
    if install_dir is None:
        install_dir = os.path.join(state.environment.get_libdir(), 'girepository-1.0')
    elif install_dir is False:
        install = False
    return TypelibTarget(typelib_output, state.subdir, state.subproject, state.environment, typelib_cmd, generated_files, [typelib_output], install=install, install_dir=[install_dir], install_tag=['typelib'], build_by_default=kwargs['build_by_default'], env=kwargs['env'])