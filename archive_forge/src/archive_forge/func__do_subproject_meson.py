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
def _do_subproject_meson(self, subp_name: str, subdir: str, default_options: T.Dict[OptionKey, str], kwargs: kwtypes.DoSubproject, ast: T.Optional[mparser.CodeBlockNode]=None, build_def_files: T.Optional[T.List[str]]=None, relaxations: T.Optional[T.Set[InterpreterRuleRelaxation]]=None) -> SubprojectHolder:
    with mlog.nested(subp_name):
        if ast:
            from ..ast import AstIndentationGenerator, AstPrinter
            printer = AstPrinter(update_ast_line_nos=True)
            ast.accept(AstIndentationGenerator())
            ast.accept(printer)
            printer.post_process()
            meson_filename = os.path.join(self.build.environment.get_build_dir(), subdir, 'meson.build')
            with open(meson_filename, 'w', encoding='utf-8') as f:
                f.write(printer.result)
            mlog.log('Generated Meson AST:', meson_filename)
            mlog.cmd_ci_include(meson_filename)
        new_build = self.build.copy()
        subi = Interpreter(new_build, self.backend, subp_name, subdir, self.subproject_dir, default_options, ast=ast, is_translated=ast is not None, relaxations=relaxations, user_defined_options=self.user_defined_options)
        subi.subprojects = self.subprojects
        subi.modules = self.modules
        subi.holder_map = self.holder_map
        subi.bound_holder_map = self.bound_holder_map
        subi.summary = self.summary
        subi.subproject_stack = self.subproject_stack + [subp_name]
        current_active = self.active_projectname
        with mlog.nested_warnings():
            subi.run()
            subi_warnings = mlog.get_warning_count()
        mlog.log('Subproject', mlog.bold(subp_name), 'finished.')
    mlog.log()
    if kwargs['version']:
        pv = subi.project_version
        wanted = kwargs['version']
        if pv == 'undefined' or not mesonlib.version_compare_many(pv, wanted)[0]:
            raise InterpreterException(f'Subproject {subp_name} version is {pv} but {wanted} required.')
    self.active_projectname = current_active
    self.subprojects.update(subi.subprojects)
    self.subprojects[subp_name] = SubprojectHolder(subi, subdir, warnings=subi_warnings, callstack=self.subproject_stack)
    if build_def_files:
        self.build_def_files.update(build_def_files)
    self.build_def_files.update(subi.build_def_files)
    self.build.merge(subi.build)
    self.build.subprojects[subp_name] = subi.project_version
    return self.subprojects[subp_name]