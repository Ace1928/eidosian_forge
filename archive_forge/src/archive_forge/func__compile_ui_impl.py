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
def _compile_ui_impl(self, state: ModuleState, kwargs: UICompilerKwArgs) -> build.GeneratedList:
    self._detect_tools(state, kwargs['method'])
    if not self.tools['uic'].found():
        err_msg = "{0} sources specified and couldn't find {1}, please check your qt{2} installation"
        raise MesonException(err_msg.format('UIC', f'uic-qt{self.qt_version}', self.qt_version))
    preserve_path_from = os.path.join(state.source_root, state.subdir) if kwargs['preserve_paths'] else None
    gen = build.Generator(self.tools['uic'], kwargs['extra_args'] + ['-o', '@OUTPUT@', '@INPUT@'], ['ui_@BASENAME@.h'], name=f'Qt{self.qt_version} ui')
    return gen.process_files(kwargs['sources'], state, preserve_path_from)