from __future__ import annotations
import os
import shlex
import subprocess
import copy
import textwrap
from pathlib import Path, PurePath
from .. import mesonlib
from .. import coredata
from .. import build
from .. import mlog
from ..modules import ModuleReturnValue, ModuleObject, ModuleState, ExtensionModule
from ..backend.backends import TestProtocol
from ..interpreterbase import (
from ..interpreter.type_checking import NoneType, ENV_KW, ENV_SEPARATOR_KW, PKGCONFIG_DEFINE_KW
from ..dependencies import Dependency, ExternalLibrary, InternalDependency
from ..programs import ExternalProgram
from ..mesonlib import HoldableObject, OptionKey, listify, Popen_safe
import typing as T
@FeatureNew('dependency.get_variable', '0.51.0')
@typed_pos_args('dependency.get_variable', optargs=[str])
@typed_kwargs('dependency.get_variable', KwargInfo('cmake', (str, NoneType)), KwargInfo('pkgconfig', (str, NoneType)), KwargInfo('configtool', (str, NoneType)), KwargInfo('internal', (str, NoneType), since='0.54.0'), KwargInfo('default_value', (str, NoneType)), PKGCONFIG_DEFINE_KW)
def variable_method(self, args: T.Tuple[T.Optional[str]], kwargs: 'kwargs.DependencyGetVariable') -> str:
    default_varname = args[0]
    if default_varname is not None:
        FeatureNew('Positional argument to dependency.get_variable()', '0.58.0').use(self.subproject, self.current_node)
    if kwargs['pkgconfig_define'] and len(kwargs['pkgconfig_define']) > 1:
        FeatureNew.single_use('dependency.get_variable keyword argument "pkgconfig_define" with more than one pair', '1.3.0', self.subproject, 'In previous versions, this silently returned a malformed value.', self.current_node)
    return self.held_object.get_variable(cmake=kwargs['cmake'] or default_varname, pkgconfig=kwargs['pkgconfig'] or default_varname, configtool=kwargs['configtool'] or default_varname, internal=kwargs['internal'] or default_varname, default_value=kwargs['default_value'], pkgconfig_define=kwargs['pkgconfig_define'])