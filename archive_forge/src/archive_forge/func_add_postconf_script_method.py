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
@typed_pos_args('meson.add_postconf_script', (str, mesonlib.File, ExternalProgram), varargs=(str, mesonlib.File, ExternalProgram))
@noKwargs
def add_postconf_script_method(self, args: T.Tuple[T.Union[str, mesonlib.File, ExternalProgram], T.List[T.Union[str, mesonlib.File, ExternalProgram]]], kwargs: 'TYPE_kwargs') -> None:
    script_args = self._process_script_args('add_postconf_script', args[1])
    script = self._find_source_script('add_postconf_script', args[0], script_args)
    self.build.postconf_scripts.append(script)