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
class ReadKwArgs(TypedDict):
    """Keyword Arguments for fs.read."""
    encoding: str