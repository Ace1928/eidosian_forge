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
def _make_mkenum_impl(self, state: 'ModuleState', sources: T.Sequence[T.Union[str, mesonlib.File, CustomTarget, CustomTargetIndex, GeneratedList]], output: str, cmd: T.List[str], *, install: bool=False, install_dir: T.Optional[T.Sequence[T.Union[str, bool]]]=None, depends: T.Optional[T.Sequence[T.Union[CustomTarget, CustomTargetIndex, BuildTarget]]]=None) -> build.CustomTarget:
    real_cmd: T.List[T.Union[str, 'ToolType']] = [self._find_tool(state, 'glib-mkenums')]
    real_cmd.extend(cmd)
    _install_dir = install_dir or state.environment.coredata.get_option(mesonlib.OptionKey('includedir'))
    assert isinstance(_install_dir, str), 'for mypy'
    return CustomTarget(output, state.subdir, state.subproject, state.environment, real_cmd, sources, [output], capture=True, install=install, install_dir=[_install_dir], install_tag=['devel'], extra_depends=depends, absolute_paths=True, description='Generating GObject enum file {}')