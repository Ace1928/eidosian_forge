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
def add_languages_for(self, args: T.List[str], required: bool, for_machine: MachineChoice) -> bool:
    args = [a.lower() for a in args]
    langs = set(self.compilers[for_machine].keys())
    langs.update(args)
    if 'vala' in langs and 'c' not in langs:
        FeatureNew.single_use('Adding Vala language without C', '0.59.0', self.subproject, location=self.current_node)
        args.append('c')
    if 'nasm' in langs:
        FeatureNew.single_use('Adding NASM language', '0.64.0', self.subproject, location=self.current_node)
    success = True
    for lang in sorted(args, key=compilers.sort_clink):
        if lang in self.compilers[for_machine]:
            continue
        machine_name = for_machine.get_lower_case_name()
        comp = self.coredata.compilers[for_machine].get(lang)
        if not comp:
            try:
                skip_sanity_check = self.should_skip_sanity_check(for_machine)
                if skip_sanity_check:
                    mlog.log('Cross compiler sanity tests disabled via the cross file.', once=True)
                comp = compilers.detect_compiler_for(self.environment, lang, for_machine, skip_sanity_check, self.subproject)
                if comp is None:
                    raise InvalidArguments(f'Tried to use unknown language "{lang}".')
            except mesonlib.MesonException:
                if not required:
                    mlog.log('Compiler for language', mlog.bold(lang), 'for the', machine_name, 'machine not found.')
                    success = False
                    continue
                else:
                    raise
        else:
            self.coredata.process_compiler_options(lang, comp, self.environment, self.subproject)
        if self.subproject:
            options = {}
            for k in comp.get_options():
                v = copy.copy(self.coredata.options[k])
                k = k.evolve(subproject=self.subproject)
                options[k] = v
            self.coredata.add_compiler_options(options, lang, for_machine, self.environment, self.subproject)
        if for_machine == MachineChoice.HOST or self.environment.is_cross_build():
            logger_fun = mlog.log
        else:
            logger_fun = mlog.debug
        logger_fun(comp.get_display_language(), 'compiler for the', machine_name, 'machine:', mlog.bold(' '.join(comp.get_exelist())), comp.get_version_string())
        if comp.linker is not None:
            logger_fun(comp.get_display_language(), 'linker for the', machine_name, 'machine:', mlog.bold(' '.join(comp.linker.get_exelist())), comp.linker.id, comp.linker.version)
        self.build.ensure_static_linker(comp)
        self.compilers[for_machine][lang] = comp
    return success