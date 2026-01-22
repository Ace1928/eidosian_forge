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
def _add_uninstalled_incdirs(self, incdirs: T.List[build.IncludeDirs], subdir: T.Optional[str]=None) -> None:
    for i in incdirs:
        curdir = i.get_curdir()
        for d in i.get_incdirs():
            path = os.path.join(curdir, d)
            self.uninstalled_incdirs.add(path)
    if subdir is not None:
        self.uninstalled_incdirs.add(subdir)