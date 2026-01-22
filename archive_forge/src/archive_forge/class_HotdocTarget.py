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
class HotdocTarget(CustomTarget):

    def __init__(self, name: str, subdir: str, subproject: str, hotdoc_conf: File, extra_extension_paths: T.Set[str], extra_assets: T.List[str], subprojects: T.List['HotdocTarget'], environment: Environment, **kwargs: T.Any):
        super().__init__(name, subdir, subproject, environment, **kwargs, absolute_paths=True)
        self.hotdoc_conf = hotdoc_conf
        self.extra_extension_paths = extra_extension_paths
        self.extra_assets = extra_assets
        self.subprojects = subprojects

    def __getstate__(self) -> dict:
        res = self.__dict__.copy()
        res['subprojects'] = []
        return res