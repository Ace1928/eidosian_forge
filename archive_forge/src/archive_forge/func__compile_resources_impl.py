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
def _compile_resources_impl(self, state: 'ModuleState', kwargs: 'ResourceCompilerKwArgs') -> T.List[build.CustomTarget]:
    self._detect_tools(state, kwargs['method'])
    if not self.tools['rcc'].found():
        err_msg = "{0} sources specified and couldn't find {1}, please check your qt{2} installation"
        raise MesonException(err_msg.format('RCC', f'rcc-qt{self.qt_version}', self.qt_version))
    targets: T.List[build.CustomTarget] = []
    DEPFILE_ARGS: T.List[str] = ['--depfile', '@DEPFILE@'] if self._rcc_supports_depfiles else []
    name = kwargs['name']
    sources: T.List['FileOrString'] = []
    for s in kwargs['sources']:
        if isinstance(s, (str, File)):
            sources.append(s)
        else:
            sources.extend(s.get_outputs())
    extra_args = kwargs['extra_args']
    if name:
        qrc_deps: T.List[File] = []
        for s in sources:
            qrc_deps.extend(self._parse_qrc_deps(state, s))
        res_target = build.CustomTarget(name, state.subdir, state.subproject, state.environment, self.tools['rcc'].get_command() + ['-name', name, '-o', '@OUTPUT@'] + extra_args + ['@INPUT@'] + DEPFILE_ARGS, sources, [f'{name}.cpp'], depend_files=qrc_deps, depfile=f'{name}.d', description='Compiling Qt resources {}')
        targets.append(res_target)
    else:
        for rcc_file in sources:
            qrc_deps = self._parse_qrc_deps(state, rcc_file)
            if isinstance(rcc_file, str):
                basename = os.path.basename(rcc_file)
            else:
                basename = os.path.basename(rcc_file.fname)
            name = f'qt{self.qt_version}-{basename.replace('.', '_')}'
            res_target = build.CustomTarget(name, state.subdir, state.subproject, state.environment, self.tools['rcc'].get_command() + ['-name', '@BASENAME@', '-o', '@OUTPUT@'] + extra_args + ['@INPUT@'] + DEPFILE_ARGS, [rcc_file], [f'{name}.cpp'], depend_files=qrc_deps, depfile=f'{name}.d', description='Compiling Qt resources {}')
            targets.append(res_target)
    return targets