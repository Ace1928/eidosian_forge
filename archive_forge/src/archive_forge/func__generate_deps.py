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
def _generate_deps(self, state: 'ModuleState', library: str, packages: T.List[str], install_dir: str) -> build.Data:
    outdir = state.environment.scratch_dir
    fname = os.path.join(outdir, library + '.deps')
    with open(fname, 'w', encoding='utf-8') as ofile:
        for package in packages:
            ofile.write(package + '\n')
    return build.Data([mesonlib.File(True, outdir, fname)], install_dir, install_dir, mesonlib.FileMode(), state.subproject)