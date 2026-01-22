from __future__ import annotations
from pathlib import Path, PurePath, PureWindowsPath
import hashlib
import os
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleInfo
from .. import mlog
from ..build import BuildTarget, CustomTarget, CustomTargetIndex, InvalidArguments
from ..interpreter.type_checking import INSTALL_KW, INSTALL_MODE_KW, INSTALL_TAG_KW, NoneType
from ..interpreterbase import FeatureNew, KwargInfo, typed_kwargs, typed_pos_args, noKwargs
from ..mesonlib import File, MesonException, has_path_sep, path_is_in_root, relpath
def _absolute_dir(self, state: 'ModuleState', arg: 'FileOrString') -> Path:
    """
        make an absolute path from a relative path, WITHOUT resolving symlinks
        """
    if isinstance(arg, File):
        return Path(arg.absolute_path(state.source_root, state.environment.get_build_dir()))
    return Path(state.source_root) / Path(state.subdir) / Path(arg).expanduser()