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
def _disable_if(self, condition: bool, message: T.Optional[str]) -> coredata.UserFeatureOption:
    if not condition:
        return copy.deepcopy(self.held_object)
    if self.value == 'enabled':
        err_msg = f'Feature {self.held_object.name} cannot be enabled'
        if message:
            err_msg += f': {message}'
        raise InterpreterException(err_msg)
    return self.as_disabled()