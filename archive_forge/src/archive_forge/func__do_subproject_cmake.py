from __future__ import annotations
from .. import mparser
from .. import environment
from .. import coredata
from .. import dependencies
from .. import mlog
from .. import build
from .. import optinterpreter
from .. import compilers
from .. import envconfig
from ..wrap import wrap, WrapMode
from .. import mesonlib
from ..mesonlib import (EnvironmentVariables, ExecutableSerialisation, MesonBugException, MesonException, HoldableObject,
from ..programs import ExternalProgram, NonExistingExternalProgram
from ..dependencies import Dependency
from ..depfile import DepFile
from ..interpreterbase import ContainerTypeInfo, InterpreterBase, KwargInfo, typed_kwargs, typed_pos_args
from ..interpreterbase import noPosargs, noKwargs, permittedKwargs, noArgsFlattening, noSecondLevelHolderResolving, unholder_return
from ..interpreterbase import InterpreterException, InvalidArguments, InvalidCode, SubdirDoneRequest
from ..interpreterbase import Disabler, disablerIfNotFound
from ..interpreterbase import FeatureNew, FeatureDeprecated, FeatureBroken, FeatureNewKwargs
from ..interpreterbase import ObjectHolder, ContextManagerObject
from ..interpreterbase import stringifyUserArguments
from ..modules import ExtensionModule, ModuleObject, MutableModuleObject, NewExtensionModule, NotFoundExtensionModule
from ..optinterpreter import optname_regex
from . import interpreterobjects as OBJ
from . import compiler as compilerOBJ
from .mesonmain import MesonMain
from .dependencyfallbacks import DependencyFallbacksHolder
from .interpreterobjects import (
from .type_checking import (
from . import primitives as P_OBJ
from pathlib import Path
from enum import Enum
import os
import shutil
import uuid
import re
import stat
import collections
import typing as T
import textwrap
import importlib
import copy
def _do_subproject_cmake(self, subp_name: str, subdir: str, default_options: T.Dict[OptionKey, str], kwargs: kwtypes.DoSubproject) -> SubprojectHolder:
    from ..cmake import CMakeInterpreter
    with mlog.nested(subp_name):
        prefix = self.coredata.options[OptionKey('prefix')].value
        from ..modules.cmake import CMakeSubprojectOptions
        options = kwargs.get('options') or CMakeSubprojectOptions()
        cmake_options = kwargs.get('cmake_options', []) + options.cmake_options
        cm_int = CMakeInterpreter(Path(subdir), Path(prefix), self.build.environment, self.backend)
        cm_int.initialise(cmake_options)
        cm_int.analyse()
        ast = cm_int.pretend_to_be_meson(options.target_options)
        result = self._do_subproject_meson(subp_name, subdir, default_options, kwargs, ast, [str(f) for f in cm_int.bs_files], relaxations={InterpreterRuleRelaxation.ALLOW_BUILD_DIR_FILE_REFERENCES})
        result.cm_interpreter = cm_int
    return result