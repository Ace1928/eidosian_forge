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
def _detect_tools(self, state: 'ModuleState', method: str, required: bool=True) -> None:
    if self._tools_detected:
        return
    self._tools_detected = True
    mlog.log(f'Detecting Qt{self.qt_version} tools')
    kwargs = {'required': required, 'modules': 'Core', 'method': method}
    qt = T.cast('QtPkgConfigDependency', find_external_dependency(f'qt{self.qt_version}', state.environment, kwargs))
    if qt.found():
        self.compilers_detect(state, qt)
        if version_compare(qt.version, '>=5.15.0'):
            self._moc_supports_depfiles = True
        else:
            mlog.warning('moc dependencies will not work properly until you move to Qt >= 5.15', fatal=False)
        if version_compare(qt.version, '>=5.14.0'):
            self._rcc_supports_depfiles = True
        else:
            mlog.warning('rcc dependencies will not work properly until you move to Qt >= 5.14:', mlog.bold('https://bugreports.qt.io/browse/QTBUG-45460'), fatal=False)
    else:
        suffix = f'-qt{self.qt_version}'
        self.tools['moc'] = NonExistingExternalProgram(name='moc' + suffix)
        self.tools['uic'] = NonExistingExternalProgram(name='uic' + suffix)
        self.tools['rcc'] = NonExistingExternalProgram(name='rcc' + suffix)
        self.tools['lrelease'] = NonExistingExternalProgram(name='lrelease' + suffix)