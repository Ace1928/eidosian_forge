from __future__ import annotations
import os
import typing as T
from .. import mesonlib
from .. import dependencies
from .. import build
from .. import mlog, coredata
from ..mesonlib import MachineChoice, OptionKey
from ..programs import OverrideProgram, ExternalProgram
from ..interpreter.type_checking import ENV_KW, ENV_METHOD_KW, ENV_SEPARATOR_KW, env_convertor_with_method
from ..interpreterbase import (MesonInterpreterObject, FeatureNew, FeatureDeprecated,
from .primitives import MesonVersionString
from .type_checking import NATIVE_KW, NoneType
@typed_pos_args('meson.add_install_script', (str, mesonlib.File, build.Executable, ExternalProgram), varargs=(str, mesonlib.File, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, ExternalProgram))
@typed_kwargs('meson.add_install_script', KwargInfo('skip_if_destdir', bool, default=False, since='0.57.0'), KwargInfo('install_tag', (str, NoneType), since='0.60.0'), KwargInfo('dry_run', bool, default=False, since='1.1.0'))
def add_install_script_method(self, args: T.Tuple[T.Union[str, mesonlib.File, build.Executable, ExternalProgram], T.List[T.Union[str, mesonlib.File, build.BuildTarget, build.CustomTarget, build.CustomTargetIndex, ExternalProgram]]], kwargs: 'AddInstallScriptKW') -> None:
    script_args = self._process_script_args('add_install_script', args[1])
    script = self._find_source_script('add_install_script', args[0], script_args)
    script.skip_if_destdir = kwargs['skip_if_destdir']
    script.tag = kwargs['install_tag']
    script.dry_run = kwargs['dry_run']
    self.build.install_scripts.append(script)