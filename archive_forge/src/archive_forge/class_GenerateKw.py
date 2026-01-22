from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
import os
import typing as T
from . import NewExtensionModule, ModuleInfo
from . import ModuleReturnValue
from .. import build
from .. import dependencies
from .. import mesonlib
from .. import mlog
from ..coredata import BUILTIN_DIR_OPTIONS
from ..dependencies.pkgconfig import PkgConfigDependency, PkgConfigInterface
from ..interpreter.type_checking import D_MODULE_VERSIONS_KW, INSTALL_DIR_KW, VARIABLES_KW, NoneType
from ..interpreterbase import FeatureNew, FeatureDeprecated
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
class GenerateKw(TypedDict):
    version: T.Optional[str]
    name: T.Optional[str]
    filebase: T.Optional[str]
    description: T.Optional[str]
    url: str
    subdirs: T.List[str]
    conflicts: T.List[str]
    dataonly: bool
    libraries: T.List[ANY_DEP]
    libraries_private: T.List[ANY_DEP]
    requires: T.List[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]
    requires_private: T.List[T.Union[str, build.StaticLibrary, build.SharedLibrary, dependencies.Dependency]]
    install_dir: T.Optional[str]
    d_module_versions: T.List[T.Union[str, int]]
    extra_cflags: T.List[str]
    variables: T.Dict[str, str]
    uninstalled_variables: T.Dict[str, str]
    unescaped_variables: T.Dict[str, str]
    unescaped_uninstalled_variables: T.Dict[str, str]