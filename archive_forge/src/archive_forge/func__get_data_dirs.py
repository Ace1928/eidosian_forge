from __future__ import annotations
from os import path
import shlex
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import build
from .. import mesonlib
from .. import mlog
from ..interpreter.type_checking import CT_BUILD_BY_DEFAULT, CT_INPUT_KW, INSTALL_TAG_KW, OUTPUT_KW, INSTALL_DIR_KW, INSTALL_KW, NoneType, in_set_validator
from ..interpreterbase import FeatureNew, InvalidArguments
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, noPosargs, typed_kwargs, typed_pos_args
from ..programs import ExternalProgram
from ..scripts.gettext import read_linguas
@staticmethod
def _get_data_dirs(state: 'ModuleState', dirs: T.Iterable[str]) -> T.List[str]:
    """Returns source directories of relative paths"""
    src_dir = path.join(state.environment.get_source_dir(), state.subdir)
    return [path.join(src_dir, d) for d in dirs]