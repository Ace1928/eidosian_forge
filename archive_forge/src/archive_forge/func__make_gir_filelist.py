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
def _make_gir_filelist(state: 'ModuleState', srcdir: str, ns: str, nsversion: str, girtargets: T.Sequence[build.BuildTarget], libsources: T.Sequence[T.Union[str, mesonlib.File, GeneratedList, CustomTarget, CustomTargetIndex]]) -> str:
    gir_filelist_dir = state.backend.get_target_private_dir_abs(girtargets[0])
    if not os.path.isdir(gir_filelist_dir):
        os.mkdir(gir_filelist_dir)
    gir_filelist_filename = os.path.join(gir_filelist_dir, f'{ns}_{nsversion}_gir_filelist')
    with open(gir_filelist_filename, 'w', encoding='utf-8') as gir_filelist:
        for s in libsources:
            if isinstance(s, (CustomTarget, CustomTargetIndex)):
                for custom_output in s.get_outputs():
                    gir_filelist.write(os.path.join(state.environment.get_build_dir(), state.backend.get_target_dir(s), custom_output) + '\n')
            elif isinstance(s, mesonlib.File):
                gir_filelist.write(s.rel_to_builddir(state.build_to_src) + '\n')
            elif isinstance(s, GeneratedList):
                for gen_src in s.get_outputs():
                    gir_filelist.write(os.path.join(srcdir, gen_src) + '\n')
            else:
                gir_filelist.write(os.path.join(srcdir, s) + '\n')
    return gir_filelist_filename