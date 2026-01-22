from __future__ import annotations
from .common import CMakeException
from .generator import parse_generator_expressions
from .. import mlog
from ..mesonlib import version_compare
import typing as T
from pathlib import Path
from functools import lru_cache
import re
import json
import textwrap
def _cmake_add_library(self, tline: CMakeTraceLine) -> None:
    args = list(tline.args)
    if 'INTERFACE' in args:
        args.remove('INTERFACE')
        if len(args) < 1:
            return self._gen_exception('add_library', 'interface library name not specified', tline)
        self.targets[args[0]] = CMakeTarget(args[0], 'INTERFACE', {}, tline=tline, imported='IMPORTED' in args)
    elif 'IMPORTED' in args:
        args.remove('IMPORTED')
        if len(args) < 2:
            return self._gen_exception('add_library', 'requires at least 2 arguments', tline)
        self.targets[args[0]] = CMakeTarget(args[0], args[1], {}, tline=tline, imported=True)
    elif 'ALIAS' in args:
        args.remove('ALIAS')
        if len(args) < 2:
            return self._gen_exception('add_library', 'requires at least 2 arguments', tline)
        self.targets[args[0]] = CMakeTarget(args[0], 'ALIAS', {'INTERFACE_LINK_LIBRARIES': [args[1]]}, tline=tline)
    elif 'OBJECT' in args:
        return self._gen_exception('add_library', 'OBJECT libraries are not supported', tline)
    else:
        self.targets[args[0]] = CMakeTarget(args[0], 'NORMAL', {}, tline=tline)