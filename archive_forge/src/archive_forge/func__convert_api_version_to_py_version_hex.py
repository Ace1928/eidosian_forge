from __future__ import annotations
import copy, json, os, shutil, re
import typing as T
from . import ExtensionModule, ModuleInfo
from .. import mesonlib
from .. import mlog
from ..coredata import UserFeatureOption
from ..build import known_shmod_kwargs, CustomTarget, CustomTargetIndex, BuildTarget, GeneratedList, StructuredSources, ExtractedObjects, SharedModule
from ..dependencies import NotFoundDependency
from ..dependencies.detect import get_dep_identifier, find_external_dependency
from ..dependencies.python import BasicPythonExternalProgram, python_factory, _PythonDependencyBase
from ..interpreter import extract_required_kwarg, permitted_dependency_kwargs, primitives as P_OBJ
from ..interpreter.interpreterobjects import _ExternalProgramHolder
from ..interpreter.type_checking import NoneType, PRESERVE_PATH_KW, SHARED_MOD_KWS
from ..interpreterbase import (
from ..mesonlib import MachineChoice, OptionKey
from ..programs import ExternalProgram, NonExistingExternalProgram
def _convert_api_version_to_py_version_hex(self, api_version: str, detected_version: str) -> str:
    python_api_version_format = re.compile('[0-9]\\.[0-9]{1,2}')
    decimal_match = python_api_version_format.fullmatch(api_version)
    if not decimal_match:
        raise InvalidArguments(f'Python API version invalid: "{api_version}".')
    if mesonlib.version_compare(api_version, '<3.2'):
        raise InvalidArguments(f'Python Limited API version invalid: {api_version} (must be greater than 3.2)')
    if mesonlib.version_compare(api_version, '>' + detected_version):
        raise InvalidArguments(f'Python Limited API version too high: {api_version} (detected {detected_version})')
    version_components = api_version.split('.')
    major = int(version_components[0])
    minor = int(version_components[1])
    return '0x{:02x}{:02x}0000'.format(major, minor)