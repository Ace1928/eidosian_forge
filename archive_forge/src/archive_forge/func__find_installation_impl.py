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
def _find_installation_impl(self, state: 'ModuleState', display_name: str, name_or_path: str, required: bool) -> MaybePythonProg:
    if not name_or_path:
        python = PythonExternalProgram('python3', mesonlib.python_command)
    else:
        tmp_python = ExternalProgram.from_entry(display_name, name_or_path)
        python = PythonExternalProgram(display_name, ext_prog=tmp_python)
        if not python.found() and mesonlib.is_windows():
            pythonpath = self._get_win_pythonpath(name_or_path)
            if pythonpath is not None:
                name_or_path = pythonpath
                python = PythonExternalProgram(name_or_path)
        if not python.found() and name_or_path in {'python2', 'python3'}:
            tmp_python = ExternalProgram.from_entry(display_name, 'python')
            python = PythonExternalProgram(name_or_path, ext_prog=tmp_python)
    if python.found():
        if python.sanity(state):
            return python
        else:
            sanitymsg = f'{python} is not a valid python or it is missing distutils'
            if required:
                raise mesonlib.MesonException(sanitymsg)
            else:
                mlog.warning(sanitymsg, location=state.current_node)
    return NonExistingExternalProgram(python.name)