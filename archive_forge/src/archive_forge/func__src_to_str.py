from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def _src_to_str(src_file: T.Union[mesonlib.FileOrString, build.CustomTarget, build.StructuredSources, build.CustomTargetIndex, build.GeneratedList]) -> T.List[str]:
    if isinstance(src_file, str):
        return [src_file]
    if isinstance(src_file, mesonlib.File):
        return [src_file.absolute_path(backend.source_dir, backend.build_dir)]
    if isinstance(src_file, (build.CustomTarget, build.CustomTargetIndex, build.GeneratedList)):
        return src_file.get_outputs()
    if isinstance(src_file, build.StructuredSources):
        return [f for s in src_file.as_list() for f in _src_to_str(s)]
    raise mesonlib.MesonBugException(f'Invalid file type {type(src_file)}.')