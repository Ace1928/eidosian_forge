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
def _get_langs_compilers_flags(state: 'ModuleState', langs_compilers: T.List[T.Tuple[str, 'Compiler']]) -> T.Tuple[T.List[str], T.List[str], T.List[str]]:
    cflags: T.List[str] = []
    internal_ldflags: T.List[str] = []
    external_ldflags: T.List[str] = []
    for lang, compiler in langs_compilers:
        if state.global_args.get(lang):
            cflags += state.global_args[lang]
        if state.project_args.get(lang):
            cflags += state.project_args[lang]
        if mesonlib.OptionKey('b_sanitize') in compiler.base_options:
            sanitize = state.environment.coredata.options[mesonlib.OptionKey('b_sanitize')].value
            cflags += compiler.sanitizer_compile_args(sanitize)
            sanitize = sanitize.split(',')
            if 'address' in sanitize:
                internal_ldflags += ['-lasan']
            if 'thread' in sanitize:
                internal_ldflags += ['-ltsan']
            if 'undefined' in sanitize:
                internal_ldflags += ['-lubsan']
    return (cflags, internal_ldflags, external_ldflags)