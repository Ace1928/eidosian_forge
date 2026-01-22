from __future__ import annotations
import os
import shutil
import typing as T
import xml.etree.ElementTree as ET
from . import ModuleReturnValue, ExtensionModule
from .. import build
from .. import coredata
from .. import mlog
from ..dependencies import find_external_dependency, Dependency, ExternalLibrary, InternalDependency
from ..mesonlib import MesonException, File, version_compare, Popen_safe
from ..interpreter import extract_required_kwarg
from ..interpreter.type_checking import INSTALL_DIR_KW, INSTALL_KW, NoneType
from ..interpreterbase import ContainerTypeInfo, FeatureDeprecated, KwargInfo, noPosargs, FeatureNew, typed_kwargs
from ..programs import NonExistingExternalProgram
class UICompilerKwArgs(TypedDict):
    """Keyword arguments for the Ui Compiler method."""
    sources: T.Sequence[T.Union[FileOrString, build.CustomTarget, build.CustomTargetIndex, build.GeneratedList]]
    extra_args: T.List[str]
    method: str
    preserve_paths: bool