from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
@noKwargs
@noPosargs
@FeatureNew('source_set.all_dependencies() method', '0.52.0')
def all_dependencies_method(self, state: ModuleState, args: T.List[TYPE_var], kwargs: TYPE_kwargs) -> T.List[dependencies.Dependency]:
    self.frozen = True
    files = self.collect(lambda x: True, True)
    return list(files.deps)