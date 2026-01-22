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
class PreprocessKwArgs(TypedDict):
    sources: T.List[FileOrString]
    moc_sources: T.List[T.Union[FileOrString, build.CustomTarget]]
    moc_headers: T.List[T.Union[FileOrString, build.CustomTarget]]
    qresources: T.List[FileOrString]
    ui_files: T.List[T.Union[FileOrString, build.CustomTarget]]
    moc_extra_arguments: T.List[str]
    rcc_extra_arguments: T.List[str]
    uic_extra_arguments: T.List[str]
    include_directories: T.List[T.Union[str, build.IncludeDirs]]
    dependencies: T.List[T.Union[Dependency, ExternalLibrary]]
    method: str
    preserve_paths: bool