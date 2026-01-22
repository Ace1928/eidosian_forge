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
@staticmethod
def _qrc_nodes(state: 'ModuleState', rcc_file: 'FileOrString') -> T.Tuple[str, T.List[str]]:
    abspath: str
    if isinstance(rcc_file, str):
        abspath = os.path.join(state.environment.source_dir, state.subdir, rcc_file)
    else:
        abspath = rcc_file.absolute_path(state.environment.source_dir, state.environment.build_dir)
    rcc_dirname = os.path.dirname(abspath)
    try:
        tree = ET.parse(abspath)
        root = tree.getroot()
        result: T.List[str] = []
        for child in root[0]:
            if child.tag != 'file':
                mlog.warning('malformed rcc file: ', os.path.join(state.subdir, str(rcc_file)))
                break
            elif child.text is None:
                raise MesonException(f'<file> element without a path in {os.path.join(state.subdir, str(rcc_file))}')
            else:
                result.append(child.text)
        return (rcc_dirname, result)
    except MesonException:
        raise
    except Exception:
        raise MesonException(f'Unable to parse resource file {abspath}')