from __future__ import annotations
import typing as T
from . import ExtensionModule, ModuleObject, MutableModuleObject, ModuleInfo
from .. import build
from .. import dependencies
from .. import mesonlib
from ..interpreterbase import (
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import OrderedSet
class AddKwargs(TypedDict):
    when: T.List[T.Union[str, dependencies.Dependency]]
    if_true: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes, dependencies.Dependency]]
    if_false: T.List[T.Union[mesonlib.FileOrString, build.GeneratedTypes]]