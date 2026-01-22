from __future__ import annotations
import os, subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build, mesonlib, mlog
from ..build import CustomTarget, CustomTargetIndex
from ..dependencies import Dependency, InternalDependency
from ..interpreterbase import (
from ..interpreter.interpreterobjects import _CustomTargetHolder
from ..interpreter.type_checking import NoneType
from ..mesonlib import File, MesonException
from ..programs import ExternalProgram
class HotdocExternalProgram(ExternalProgram):

    def run_hotdoc(self, cmd: T.List[str]) -> int:
        return subprocess.run(self.get_command() + cmd, stdout=subprocess.DEVNULL).returncode