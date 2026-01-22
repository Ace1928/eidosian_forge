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
def _check_generated_pc_deprecation(self, obj: T.Union[build.CustomTarget, build.CustomTargetIndex, build.StaticLibrary, build.SharedLibrary]) -> None:
    if obj.get_id() in self.metadata:
        return
    data = self.metadata[obj.get_id()]
    if data.warned:
        return
    mlog.deprecation('Library', mlog.bold(obj.name), 'was passed to the "libraries" keyword argument of a previous call to generate() method instead of first positional argument.', 'Adding', mlog.bold(data.display_name), 'to "Requires" field, but this is a deprecated behaviour that will change in a future version of Meson. Please report the issue if this warning cannot be avoided in your case.', location=data.location)
    data.warned = True