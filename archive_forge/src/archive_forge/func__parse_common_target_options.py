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
def _parse_common_target_options(self, func: str, private_prop: str, interface_prop: str, tline: CMakeTraceLine, ignore: T.Optional[T.List[str]]=None, paths: bool=False) -> None:
    if ignore is None:
        ignore = ['BEFORE']
    args = list(tline.args)
    if len(args) < 1:
        return self._gen_exception(func, 'requires at least one argument', tline)
    target = args[0]
    if target not in self.targets:
        return self._gen_exception(func, f'TARGET {target} not found', tline)
    interface = []
    private = []
    mode = 'PUBLIC'
    for i in args[1:]:
        if i in ignore:
            continue
        if i in {'INTERFACE', 'LINK_INTERFACE_LIBRARIES', 'PUBLIC', 'PRIVATE', 'LINK_PUBLIC', 'LINK_PRIVATE'}:
            mode = i
            continue
        if mode in {'INTERFACE', 'LINK_INTERFACE_LIBRARIES', 'PUBLIC', 'LINK_PUBLIC'}:
            interface += i.split(';')
        if mode in {'PUBLIC', 'PRIVATE', 'LINK_PRIVATE'}:
            private += i.split(';')
    if paths:
        interface = self._guess_files(interface)
        private = self._guess_files(private)
    interface = [x for x in interface if x]
    private = [x for x in private if x]
    for j in [(private_prop, private), (interface_prop, interface)]:
        if not j[0] in self.targets[target].properties:
            self.targets[target].properties[j[0]] = []
        self.targets[target].properties[j[0]] += j[1]